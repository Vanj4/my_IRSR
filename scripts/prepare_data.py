"""仅负责原有PyTorch bicubic在线退化，不读取文件、不裁剪、不做增强。
算法保持：align_corners=False、antialias=True，输出clamp到[0,1]。
倍率由dataset传入；省略时读取config.SCALE。没有切换成MATLAB风格实现。
"""
import torch
from torch.nn import functional as F
import config

def _validate_scale(scale: int) -> None:
    if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
        raise ValueError("scale 必须是正整数")

def bicubic_degradation(hr: torch.Tensor, scale: int | None = None) -> torch.Tensor:
    """将 [C,H,W] 或 [N,C,H,W] 的 HR 在线缩小，返回同阶 LR 张量。

    输入必须是 [0, 1] 浮点张量，空间尺寸必须能被 scale 整除。
    本函数不自动裁剪 HR，防止调用方丢失 LR/HR 对齐关系。
    """
    scale = config.SCALE if scale is None else scale
    _validate_scale(scale)
    if hr.ndim not in (3, 4) or not hr.is_floating_point():
        raise ValueError("hr 必须是 [C,H,W] 或 [N,C,H,W] 浮点张量")
    if any(size == 0 for size in hr.shape):
        raise ValueError("hr 各维度必须非空")
    height, width = hr.shape[-2:]
    if height % scale or width % scale:
        raise ValueError("HR 高度和宽度必须能被 scale 整除")
    if not torch.isfinite(hr).all() or hr.min() < 0 or hr.max() > 1:
        raise ValueError("hr 像素必须是 [0, 1] 内的有限值")
    batched = hr.unsqueeze(0) if hr.ndim == 3 else hr
    lr = F.interpolate(batched, size=(height // scale, width // scale),
                       mode="bicubic", align_corners=False, antialias=True)
    lr = lr.clamp(0.0, 1.0)
    return lr.squeeze(0) if hr.ndim == 3 else lr
