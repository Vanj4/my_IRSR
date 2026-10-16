"""独立RGB高频编码器：不抽取Haar，保持HR空间尺寸；供RGB引导SR主干调用。"""
import torch
from torch import nn
from torch.nn import functional as F


class RGBHighFrequencyEncoder(nn.Module):
    """输入[B,3,H,W]的[0,1] RGB，输出[B,out_channels,H,W]。

    使用stride=1的不抽取Haar滤波，不是stride=2的临界采样DWT。
    LH表示行方向低通、列方向高通；HL反之。
    """

    def __init__(self, out_channels: int = 96):
        super().__init__()
        if isinstance(out_channels, bool) or not isinstance(out_channels, int) or out_channels < 1:
            raise ValueError("out_channels必须为正整数")

        # 二维Haar滤波器：一维滤波器的外积，因此归一化因子为1/2。
        # 每个RGB通道分别使用LL/LH/HL/HH四个固定滤波器。
        filters = torch.tensor([
            [[1., 1.], [1., 1.]],      # LL：两个方向均低通
            [[1., -1.], [1., -1.]],    # LH：列方向差分
            [[1., 1.], [-1., -1.]],    # HL：行方向差分
            [[1., -1.], [-1., 1.]],    # HH：对角高频
        ]).unsqueeze(1) * 0.5
        # [12,1,2,2]，配合groups=3，禁止滤波阶段混合RGB通道。
        self.register_buffer("haar_filters", filters.repeat(3, 1, 1, 1))

        # 三个高频子带×RGB三个通道=9通道。
        self.encoder = nn.Sequential(
            nn.Conv2d(9, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        # 对高频输入做通道投影，作为独立残差旁路。
        self.input_projection = nn.Conv2d(9, out_channels, kernel_size=1)

    def haar_subbands(self, rgb: torch.Tensor):
        """返回LL/LH/HL/HH，每个均为[B,3,H,W]，不下采样。"""
        if (rgb.ndim != 4 or rgb.shape[1] != 3 or min(rgb.shape) < 1
                or not rgb.is_floating_point()):
            raise ValueError("输入必须为非空浮点RGB张量[B,3,H,W]")
        b, _, h, w = rgb.shape
        # 在FP32中计算半精度输入的相减，减轻弱高频的消减误差。
        # float64输入保留float64，便于数值验证。
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            work = rgb.float() if rgb.dtype in (torch.float16, torch.bfloat16) else rgb
            # 2×2核配合右/下各补1像素，stride=1保持H、W不变。
            mode = "reflect" if min(h, w) > 1 else "replicate"
            padded = F.pad(work, (0, 1, 0, 1), mode=mode)
            bands = F.conv2d(
                padded,
                self.haar_filters.to(device=work.device, dtype=work.dtype),
                stride=1,
                groups=3,
            )
            # 分组卷积的通道顺序：R四带、G四带、B四带。
            bands = bands.reshape(b, 3, 4, h, w).to(dtype=rgb.dtype)
        return bands.unbind(dim=2)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        ll, lh, hl, hh = self.haar_subbands(rgb)
        # 本分支不使用LL；拼接顺序为LH_RGB、HL_RGB、HH_RGB。
        high = torch.cat((lh, hl, hh), dim=1)  # [B,9,H,W]
        return self.encoder(high) + self.input_projection(high)


class RGBGuidanceEncoder(nn.Module):
    """HR高频编码后逐级压缩8倍；推理可用带halo的等价局部分块。"""
    def __init__(self, channels=96, inference_tile=256):
        super().__init__()
        if inference_tile < 0 or inference_tile % 8:
            raise ValueError("RGB编码tile必须为0或8的正整数倍")
        self.high_encoder = RGBHighFrequencyEncoder(channels)
        self.downsample = nn.Sequential(*[
            layer for _ in range(3) for layer in (
                nn.Conv2d(channels, channels, 3, stride=2, padding=1),
                nn.GELU(),
            )
        ])
        self.inference_tile = inference_tile

    def _encode(self, rgb):
        return self.downsample(self.high_encoder(rgb))

    def forward(self, rgb):
        h, w = rgb.shape[-2:]
        if h % 8 or w % 8:
            raise ValueError("RGB引导编码输入宽高必须为8的倍数")
        tile = self.inference_tile
        if self.training or not tile or max(h, w) <= tile:
            return self._encode(rgb)
        # 整个局部编码器的HR依赖偏移为[-9,+10]；16像素halo足够。
        # tile、halo及裁片原点均对齐8，保证stride=2三次的采样相位一致。
        # 只切分局部RGB编码；无窗口注意力/全局统计，不做重叠加权混合。
        halo = 16
        rows = []
        for y in range(0, h, tile):
            cols = []
            y1 = min(y + tile, h)
            sy, ey = max(0, y - halo), min(h, y1 + halo)
            for x in range(0, w, tile):
                x1 = min(x + tile, w)
                sx, ex = max(0, x - halo), min(w, x1 + halo)
                encoded = self._encode(rgb[..., sy:ey, sx:ex])
                oy, ox = (y - sy) // 8, (x - sx) // 8
                cols.append(encoded[..., oy:oy+(y1-y)//8, ox:ox+(x1-x)//8])
            rows.append(torch.cat(cols, dim=-1))
        return torch.cat(rows, dim=-2)
