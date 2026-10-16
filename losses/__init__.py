"""红外图像超分辨率训练损失。"""

from .sr_loss import EdgeLoss, GradientLoss, L1Loss, SRLoss

__all__ = ["L1Loss", "GradientLoss", "EdgeLoss", "SRLoss"]
