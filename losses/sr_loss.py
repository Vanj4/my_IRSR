"""红外超分辨率损失，输入均为浮点张量 [N, C, H, W]。

总损失 = l1_weight * L1 + gradient_weight * 梯度损失
         + edge_weight * 边缘损失。
默认权重为 1.0、0.0、0.0，即仅启用像素 L1。
SR 和 HR 应使用相同的数值范围；本模块不进行归一化或裁剪。

用法：
    from losses import SRLoss
    criterion = SRLoss(gradient_weight=0.0, edge_weight=0.0)
    loss = criterion(sr, hr)
    loss.backward()
"""

import math
import config

import torch
from torch import nn
from torch.nn import functional as F


def _validate_images(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.ndim != 4 or pred.shape != target.shape:
        raise ValueError("pred 和 target 必须为形状相同的 [N, C, H, W] 张量")
    if any(size == 0 for size in pred.shape):
        raise ValueError("输入张量的各维度必须非空")
    if not pred.is_floating_point() or not target.is_floating_point():
        raise TypeError("pred 和 target 必须为浮点张量")
    if pred.device != target.device:
        raise ValueError("pred 和 target 必须位于同一设备")


class L1Loss(nn.Module):
    """像素绝对误差，对 batch、通道和空间维度取均值。"""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _validate_images(pred, target)
        return F.l1_loss(pred, target)


class GradientLoss(nn.Module):
    """水平/垂直一阶前向差分之间的 L1，两个方向等权平均。

    L_grad = (mean(|dx_pred - dx_target|)
              + mean(|dy_pred - dy_target|)) / 2。
    不进行边界填充，要求 H、W 均至少为 2。
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _validate_images(pred, target)
        if min(pred.shape[-2:]) < 2:
            raise ValueError("梯度损失要求图像高度和宽度均至少为 2")
        pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
        target_dx = target[..., :, 1:] - target[..., :, :-1]
        pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
        target_dy = target[..., 1:, :] - target[..., :-1, :]
        return 0.5 * (F.l1_loss(pred_dx, target_dx)
                      + F.l1_loss(pred_dy, target_dy))


class EdgeLoss(nn.Module):
    """Sobel 水平/垂直边缘响应之间的 L1，两个方向等权平均。

    使用除以 8 的 3x3 Sobel 核和 replicate 边界填充，逐通道计算；
    保留有符号响应，不使用不可微的二值边缘检测或阈值操作。
    与 GradientLoss 的相邻像素差分相比，Sobel 在正交方向进行平滑。
    """

    def __init__(self) -> None:
        super().__init__()
        kernel_x = torch.tensor([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ]) / 8.0
        self.register_buffer("kernels", torch.stack(
            (kernel_x, kernel_x.t()), dim=0
        ).unsqueeze(1))

    def _edges(self, image: torch.Tensor) -> torch.Tensor:
        # 分组卷积为每个输入通道生成 dx、dy 两个响应图。
        channels = image.shape[1]
        kernels = self.kernels.to(device=image.device, dtype=image.dtype)
        kernels = kernels.repeat(channels, 1, 1, 1)
        padded = F.pad(image, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(padded, kernels, groups=channels)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _validate_images(pred, target)
        return F.l1_loss(self._edges(pred), self._edges(target))


class SRLoss(nn.Module):
    """可配置的总损失；权重为 0 的项跳过计算，不影响训练开销。"""

    def __init__(self, l1_weight: float | None = None,
                 gradient_weight: float | None = None,
                 edge_weight: float | None = None) -> None:
        super().__init__()
        l1_weight = config.L1_WEIGHT if l1_weight is None else l1_weight
        gradient_weight = config.GRADIENT_WEIGHT if gradient_weight is None else gradient_weight
        edge_weight = config.EDGE_WEIGHT if edge_weight is None else edge_weight
        for name, value in (("l1_weight", l1_weight),
                            ("gradient_weight", gradient_weight),
                            ("edge_weight", edge_weight)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须是有限的非负数")
            setattr(self, name, float(value))
        self.l1 = L1Loss()
        self.gradient = GradientLoss()
        self.edge = EdgeLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _validate_images(pred, target)
        # 保持全零权重时也可反向传播，且不扫描整幅图像计算无效项。
        total = pred.reshape(-1)[0] * 0.0
        if self.l1_weight:
            total = total + self.l1_weight * self.l1(pred, target)
        if self.gradient_weight:
            total = total + self.gradient_weight * self.gradient(pred, target)
        if self.edge_weight:
            total = total + self.edge_weight * self.edge(pred, target)
        return total
