"""Native-LR infrared SR with Haar subbands and optional HR-RGB guidance.
Input B1HW in [0,1]; output B1(4H)(4W), NOT clamped during training.
Feature subbands are not image-domain HR wavelet coefficients.
"""
import math
import torch
import config
from rgb_high_frequency_encoder import RGBGuidanceEncoder
from spatial_transformer import SpatialRefinement
from torch import nn
from torch.nn import functional as F


def haar_dwt(x):
    """Orthonormal Haar, order LL, row-high, column-high, diagonal-high."""
    if x.ndim != 4 or x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError('DWT expects BCHW with even H/W')
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return ((a+b+c+d)*0.5, (a+b-c-d)*0.5,
            (a-b+c-d)*0.5, (a-b-c+d)*0.5)


def haar_iwt(bands):
    if len(bands) != 4 or any(t.shape != bands[0].shape for t in bands):
        raise ValueError('IWT needs four equally shaped subbands')
    ll, hr, hc, hd = bands
    a, b = (ll+hr+hc+hd)*0.5, (ll+hr-hc-hd)*0.5
    c, d = (ll-hr+hc-hd)*0.5, (ll-hr-hc+hd)*0.5
    top = torch.stack((a,b), dim=-1).flatten(-2)
    bottom = torch.stack((c,d), dim=-1).flatten(-2)
    return torch.stack((top,bottom), dim=-2).flatten(-3,-2)


class ConvEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels,channels,3,padding=1),
                                  nn.GELU(), nn.Conv2d(channels,channels,3,padding=1))

    def forward(self, x):
        return x + self.body(x)


class LowToHighAttention(nn.Module):
    """Q from directional HF, K/V from LL; attention within aligned windows.
    The same HF encoder/attention weights are shared by all three directions.
    """
    def __init__(self, channels=None, heads=None, window_size=None):
        super().__init__()
        channels = config.CHANNELS if channels is None else channels
        heads = config.ATTENTION_HEADS if heads is None else heads
        window_size = config.WINDOW_SIZE if window_size is None else window_size
        if channels % heads:
            raise ValueError('channels must be divisible by heads')
        self.channels, self.heads, self.window_size = channels, heads, window_size
        self.norm_high, self.norm_low = nn.LayerNorm(channels), nn.LayerNorm(channels)
        self.q = nn.Linear(channels,channels)
        self.kv = nn.Linear(channels,2*channels)
        self.proj = nn.Linear(channels,channels)
        self.alpha = nn.Parameter(torch.tensor(float(config.CROSS_ALPHA_INIT)))
        w = window_size
        self.relative_bias = nn.Parameter(torch.zeros((2*w-1)**2,heads))
        coords = torch.stack(torch.meshgrid(torch.arange(w),torch.arange(w),indexing='ij')).flatten(1)
        delta = coords[:,:,None] - coords[:,None,:]
        index = (delta[0]+w-1)*(2*w-1) + delta[1]+w-1
        self.register_buffer('relative_index',index.long(),persistent=False)

    def partition(self,x):
        b,c,h,w = x.shape
        s = self.window_size
        if h%s or w%s:
            raise ValueError('Subband size must be a window multiple')
        return x.reshape(b,c,h//s,s,w//s,s).permute(0,2,4,3,5,1).reshape(-1,s*s,c)

    def reverse(self,x,shape):
        b,c,h,w=shape
        s=self.window_size
        return x.reshape(b,h//s,w//s,s,s,c).permute(0,5,1,3,2,4).reshape(b,c,h,w)

    def compute_updates(self,low,high):
        if len(high)!=3 or any(x.shape!=low.shape for x in high):
            raise ValueError('Expected LL and three matching directional features')
        lt=self.norm_low(self.partition(low))
        ht=self.norm_high(self.partition(torch.cat(high,dim=0)))
        n,t,c=lt.shape
        k,v=self.kv(lt).reshape(n,t,2,self.heads,c//self.heads).permute(2,0,3,1,4).unbind(0)
        q=self.q(ht).reshape(3*n,t,self.heads,c//self.heads).transpose(1,2)
        bias=self.relative_bias[self.relative_index].permute(2,0,1).unsqueeze(0).to(q.dtype)
        # SDPA chooses an available backend. No claim that a fused kernel always runs.
        out=F.scaled_dot_product_attention(q,k.repeat(3,1,1,1),v.repeat(3,1,1,1),
                                          attn_mask=bias,dropout_p=0.0)
        out=self.proj(out.transpose(1,2).reshape(3*n,t,c))
        spatial=self.reverse(out,(3*low.shape[0],*low.shape[1:]))
        return spatial.chunk(3,dim=0)

    def forward(self,low,high):
        return tuple(h+self.alpha*u for h,u in zip(high,self.compute_updates(low,high)))


class RGBToHighAttention(LowToHighAttention):
    """独立于IR跨频注意力：Q来自IR高频，K/V来自RGB高频引导。"""
    def __init__(self,channels,heads,window_size,gate_init=0.01):
        super().__init__(channels,heads,window_size)
        del self.alpha
        # 三个方向分别控制注入强度；共享Q/K/V和位置偏置。
        self.gamma=nn.Parameter(torch.full((3,),float(gate_init)))

    def forward(self,rgb_guide,high):
        updates=self.compute_updates(rgb_guide,high)
        return tuple(h+self.gamma[i]*u for i,(h,u) in enumerate(zip(high,updates)))


class FrequencyInteractionBlock(nn.Module):
    def __init__(self,channels,heads,window_size,use_rgb=False):
        super().__init__()
        self.low_encoder=ConvEncoder(channels)
        self.high_encoder=ConvEncoder(channels)
        self.cross_attention=LowToHighAttention(channels,heads,window_size)
        self.fuse=nn.Conv2d(channels,channels,3,padding=1)
        self.rgb_attention=(RGBToHighAttention(channels,heads,window_size,config.RGB_GATE_INIT)
                            if use_rgb else None)

    def forward(self,x,rgb_guide=None):
        low,*high=haar_dwt(x)
        low=self.low_encoder(low)
        high=self.high_encoder(torch.cat(high,dim=0)).chunk(3,dim=0)
        high=self.cross_attention(low,high)
        if self.rgb_attention is not None:
            if rgb_guide is None: raise ValueError("RGB引导已启用，但缺少RGB特征")
            high=self.rgb_attention(rgb_guide,high)
        return x+self.fuse(haar_iwt((low,*high)))


class FrequencyCrossAttentionSR(nn.Module):
    """x4 IR SR with optional HR-RGB high-frequency window guidance."""
    def __init__(self,channels=None,num_blocks=None,heads=None,window_size=None,scale=None,use_rgb=None,spatial_groups=None):
        super().__init__()
        channels = config.CHANNELS if channels is None else channels
        num_blocks = config.NUM_BLOCKS if num_blocks is None else num_blocks
        heads = config.ATTENTION_HEADS if heads is None else heads
        window_size = config.WINDOW_SIZE if window_size is None else window_size
        scale = config.SCALE if scale is None else scale
        for name,value in [('channels',channels),('num_blocks',num_blocks),('heads',heads),('window_size',window_size)]:
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError(f'{name} must be a positive integer')
        if scale!=4:
            raise ValueError('V1 reconstruction head implements x4 only')
        if channels%heads:
            raise ValueError('channels must be divisible by heads')
        self.use_rgb=config.USE_RGB_GUIDANCE if use_rgb is None else bool(use_rgb)
        self.scale,self.input_multiple=scale,2*window_size
        self.stem=nn.Conv2d(1,channels,3,padding=1)
        self.blocks=nn.Sequential(*(FrequencyInteractionBlock(channels,heads,window_size,False)
                                    for _ in range(num_blocks)))
        self.body_conv=nn.Conv2d(channels,channels,3,padding=1)
        self.upsample=nn.Sequential(nn.Conv2d(channels,4*channels,3,padding=1),nn.PixelShuffle(2),
                                   nn.Conv2d(channels,4*channels,3,padding=1),nn.PixelShuffle(2))
        self.head=nn.Conv2d(channels,1,3,padding=1)
        # 先完成全部原IR层初始化，再创建RGB层；同种子保留原IR初始权重。
        if self.use_rgb:
            for block in self.blocks:
                block.rgb_attention=RGBToHighAttention(channels,heads,window_size,config.RGB_GATE_INIT)
        self.rgb_encoder=(RGBGuidanceEncoder(channels,config.RGB_GUIDE_TILE) if self.use_rgb else None)
        groups=config.SPATIAL_GROUPS if spatial_groups is None else spatial_groups
        if isinstance(groups,bool) or not isinstance(groups,int) or groups<0:
            raise ValueError('spatial_groups must be a nonnegative integer')
        # Initialize new layers last, preserving original IR/RGB initial weights.
        self.spatial_refinement=(SpatialRefinement(channels,groups,config.SPATIAL_DEPTH,
            config.SPATIAL_HEADS,config.SPATIAL_WINDOW,config.SPATIAL_MLP_RATIO,
            config.SPATIAL_RESIDUAL_SCALE,config.SPATIAL_CHECKPOINT,config.SPATIAL_EVAL_CHUNK)
            if groups else nn.Identity())
        if groups: self.input_multiple=math.lcm(self.input_multiple,config.SPATIAL_WINDOW)


    def forward(self,lr,rgb=None):
        if lr.ndim!=4 or lr.shape[1]!=1 or min(lr.shape)<1 or not lr.is_floating_point():
            raise ValueError('Input must be nonempty floating B1HW native LR')
        h,w=lr.shape[-2:]
        ph,pw=(-h)%self.input_multiple,(-w)%self.input_multiple
        x=F.pad(lr,(0,pw,0,ph),mode='reflect' if ph<h and pw<w else 'replicate')
        rgb_guide=None
        if self.use_rgb:
            if (rgb is None or rgb.ndim!=4 or rgb.shape[:2]!=(lr.shape[0],3)
                    or rgb.shape[-2:]!=(h*self.scale,w*self.scale)
                    or rgb.device!=lr.device or not rgb.is_floating_point()):
                raise ValueError("RGB引导需要同批次、同设备的[B,3,4H,4W]浮点RGB")
            # IR在LR网格补边，RGB在对应HR网格补同一范围；禁止任意resize。
            rh,rw=rgb.shape[-2:]
            rgb=F.pad(rgb,(0,pw*self.scale,0,ph*self.scale),
                      mode='reflect' if ph*self.scale<rh and pw*self.scale<rw else 'replicate')
            rgb_guide=self.rgb_encoder(rgb)
            if rgb_guide.shape[-2:]!=(x.shape[-2]//2,x.shape[-1]//2):
                raise ValueError("RGB引导和IR子带空间尺寸不匹配")
        shallow=self.stem(x)
        deep=shallow
        for block in self.blocks:
            deep=block(deep,rgb_guide)
        deep=self.spatial_refinement(deep)
        features=shallow+self.body_conv(deep)
        residual=self.head(self.upsample(features))[...,:h*self.scale,:w*self.scale]
        # Baseline on ORIGINAL LR, independent of internal wavelet/window padding.
        with torch.autocast(device_type=lr.device.type,enabled=False):
            baseline=F.interpolate(lr.float(),scale_factor=self.scale,mode='bicubic',
                                   align_corners=False,antialias=True).clamp(0,1)
        return baseline+residual.float()
