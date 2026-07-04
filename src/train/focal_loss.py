"""
Focal Loss 직접 구현.

Reference:
    Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
    https://arxiv.org/abs/1708.02002

수식:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

gamma=0 일 때 가중치 적용 BCE와 동일하게 동작함을 보장.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-label Binary Focal Loss.

    Args:
        gamma     : Focusing parameter. 0 = weighted BCE, 2 권장.
        pos_weight: 양성 클래스 가중치 Tensor shape (num_classes,). None이면 균등.
        reduction : 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : Raw model outputs (before sigmoid), shape (B, C)
            targets: Binary labels, shape (B, C), values ∈ {0, 1}

        Returns:
            Scalar loss value
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma
        focal_loss   = focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

    def __repr__(self) -> str:
        return (
            f"FocalLoss(gamma={self.gamma}, "
            f"pos_weight={'set' if self.pos_weight is not None else 'None'}, "
            f"reduction='{self.reduction}')"
        )


def build_loss(
    gamma: float = 2.0,
    pos_weight: torch.Tensor | None = None,
) -> FocalLoss:
    """
    Focal Loss 팩토리 함수.

    Args:
        gamma     : Focal gamma (0=BCE, 1, 2)
        pos_weight: 클래스별 양성 가중치 (data_loader.compute_pos_weight 결과)

    Returns:
        FocalLoss 인스턴스
    """
    return FocalLoss(gamma=gamma, pos_weight=pos_weight)
