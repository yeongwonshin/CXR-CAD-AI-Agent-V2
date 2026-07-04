"""
Ensemble & Test-Time Augmentation (TTA) 모듈.

  - SoftVotingEnsemble: 여러 CAD 모델의 확률 평균 앙상블
  - TTAWrapper        : TTA 적용 래퍼 (원본+좌우반전+회전 4가지)
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from src.train.models import NUM_CLASSES


class SoftVotingEnsemble(nn.Module):
    """
    여러 CAD 모델의 Soft Voting 앙상블.

    각 모델의 예측 확률 (가중) 평균을 최종 예측으로 사용.

    Args:
        models_list: CAD 모델 리스트
        weights    : 각 모델의 가중치. None이면 균등 평균.
    """

    def __init__(
        self,
        models_list: List[nn.Module],
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.models = nn.ModuleList(models_list)

        if weights is not None:
            assert len(weights) == len(models_list)
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            self.weights = [1.0 / len(models_list)] * len(models_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """각 모델의 logits → sigmoid 확률 → 가중 평균."""
        out = torch.zeros(x.size(0), NUM_CLASSES, device=x.device, dtype=x.dtype)
        for model, w in zip(self.models, self.weights):
            out += w * torch.sigmoid(model(x))  # logits → 확률 후 평균
        return out  # (B, 14) 확률 값


class TTAWrapper(nn.Module):
    """
    Test-Time Augmentation 래퍼.

    여러 augmentation view에 대해 forward를 실행하고 확률 평균 반환.

    Args:
        model         : 기반 CAD 모델
        tta_transforms: src.preprocess.transforms.get_tta_transforms() 반환 리스트
    """

    def __init__(self, model: nn.Module, tta_transforms: list):
        super().__init__()
        self.model          = model
        self.tta_transforms = tta_transforms

    def forward(self, images_pil: list) -> torch.Tensor:
        """
        Args:
            images_pil: PIL Image 리스트 (배치)

        Returns:
            Tensor (B, 14) — TTA averaged probabilities
        """
        device    = next(self.model.parameters()).device
        all_probs = []

        for transform in self.tta_transforms:
            batch = torch.stack([transform(img) for img in images_pil]).to(device)
            with torch.no_grad():
                all_probs.append(self.model(batch))

        return torch.stack(all_probs).mean(dim=0)
