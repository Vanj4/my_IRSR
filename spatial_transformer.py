"""Spatial refinement on the native LR grid, after frequency interaction.
Swin-style regular/shifted window self-attention + MLP + group convolution.
This is a small adaptation, not a reproduction of HAT/HiT-SR/PFT.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def partition(x, size):
    b,h,w,c=x.shape
    return x.reshape(b,h//size,size,w//size,size,c).permute(0,1,3,2,4,5).reshape(-1,size*size,c)


def reverse(x, size, b, h, w):
    return x.reshape(b,h//size,w//size,size,size,-1).permute(0,1,3,2,4,5).reshape(b,h,w,-1)


class SpatialTransformerLayer(nn.Module):
    """Pre-LN self-attention: Q/K/V all come from reconstructed IR spatial features."""
    def __init__(self, channels, heads, window, shift, mlp_ratio=2.0, eval_chunk=128):
        super().__init__()
        self.heads,self.window,self.shift,self.eval_chunk=heads,window,shift,eval_chunk
        self.norm1=nn.LayerNorm(channels)
        self.qkv=nn.Linear(channels,3*channels)
        self.proj=nn.Linear(channels,channels)
        self.norm2=nn.LayerNorm(channels)
        hidden=int(channels*mlp_ratio)
        self.mlp=nn.Sequential(nn.Linear(channels,hidden),nn.GELU(),nn.Linear(hidden,channels))
        self.relative_bias=nn.Parameter(torch.zeros((2*window-1)**2,heads))
        coords=torch.stack(torch.meshgrid(torch.arange(window),torch.arange(window),indexing='ij')).flatten(1)
        delta=coords[:,:,None]-coords[:,None,:]
        self.register_buffer('relative_index',((delta[0]+window-1)*(2*window-1)+delta[1]+window-1).long(),persistent=False)
        nn.init.trunc_normal_(self.relative_bias,std=0.02)

    def region_labels(self,h,w,sh,sw,device):
        # Standard shifted-window mask: prevent cyclic wraparound across image boundaries.
        labels=torch.zeros((1,h,w,1),device=device,dtype=torch.int64)
        hs=(slice(0,-self.window),slice(-self.window,-sh),slice(-sh,None)) if sh else (slice(None),)
        ws=(slice(0,-self.window),slice(-self.window,-sw),slice(-sw,None)) if sw else (slice(None),)
        i=0
        for a in hs:
            for b in ws:
                labels[:,a,b,:]=i
                i+=1
        return partition(labels,self.window).squeeze(-1)

    def forward(self,x):
        b,c,h,w=x.shape
        s=self.window
        if h%s or w%s: raise ValueError('Spatial transformer requires window-aligned features')
        sh=self.shift if h>s else 0
        sw=self.shift if w>s else 0
        shortcut=x.permute(0,2,3,1)
        z=self.norm1(shortcut)
        if sh or sw: z=torch.roll(z,(-sh,-sw),(1,2))
        tokens=partition(z,s)
        n,t,_=tokens.shape
        q,k,v=self.qkv(tokens).reshape(n,t,3,self.heads,c//self.heads).permute(2,0,3,1,4).unbind(0)
        bias=self.relative_bias[self.relative_index].permute(2,0,1).unsqueeze(0).to(q.dtype)
        labels=self.region_labels(h,w,sh,sw,x.device) if sh or sw else None
        # Chunk independent attention windows only at inference. No spatial tiling/seams.
        chunk=self.eval_chunk if not self.training and self.eval_chunk else n
        outputs=[]
        for start in range(0,n,chunk):
            end=min(start+chunk,n)
            mask=bias
            if labels is not None:
                ids=torch.arange(start,end,device=x.device)%labels.shape[0]
                regions=labels[ids]
                blocked=regions[:,:,None]!=regions[:,None,:]
                mask=bias.expand(end-start,-1,-1,-1).masked_fill(blocked[:,None],float('-inf'))
            outputs.append(F.scaled_dot_product_attention(q[start:end],k[start:end],v[start:end],attn_mask=mask,dropout_p=0.0))
        out=torch.cat(outputs,0).transpose(1,2).reshape(n,t,c)
        out=reverse(self.proj(out),s,b,h,w)
        if sh or sw: out=torch.roll(out,(sh,sw),(1,2))
        out=shortcut+out
        out=out+self.mlp(self.norm2(out))
        return out.permute(0,3,1,2).contiguous()


class ResidualSpatialTransformerGroup(nn.Module):
    def __init__(self,channels,heads,window,depth,mlp_ratio,residual_scale,use_checkpoint,eval_chunk):
        super().__init__()
        self.layers=nn.ModuleList(SpatialTransformerLayer(channels,heads,window,0 if i%2==0 else window//2,mlp_ratio,eval_chunk) for i in range(depth))
        self.conv=nn.Conv2d(channels,channels,3,padding=1)
        self.residual_scale=residual_scale
        self.use_checkpoint=use_checkpoint

    def forward(self,x):
        z=x
        for layer in self.layers:
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                z=checkpoint(layer,z,use_reentrant=False)
            else:
                z=layer(z)
        # Fixed moderate residual scale, not a learnable near-zero gate.
        return x+self.residual_scale*self.conv(z)


class SpatialRefinement(nn.Sequential):
    def __init__(self,channels,groups,depth,heads,window,mlp_ratio,residual_scale,use_checkpoint,eval_chunk):
        for name,value in [('groups',groups),('depth',depth),('heads',heads),('window',window),('eval_chunk',eval_chunk)]:
            if isinstance(value,bool) or not isinstance(value,int) or value<1: raise ValueError(f'{name} must be a positive integer')
        if channels%heads: raise ValueError('channels must be divisible by spatial heads')
        if window%2 or depth%2: raise ValueError('Use even window and even depth for paired regular/shifted layers')
        if not math.isfinite(mlp_ratio) or mlp_ratio<=0 or int(channels*mlp_ratio)<1: raise ValueError('Invalid MLP ratio')
        if not math.isfinite(residual_scale) or residual_scale<=0: raise ValueError('Invalid group residual scale')
        super().__init__(*(ResidualSpatialTransformerGroup(channels,heads,window,depth,mlp_ratio,residual_scale,use_checkpoint,eval_chunk) for _ in range(groups)))
