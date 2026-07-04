"""
모델 Calibration 분석.

- 신뢰도 곡선 (Reliability Diagram)
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Temperature Scaling
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE) 계산.

    Args:
        y_true: (N, C) or (N,) binary labels
        y_prob: (N, C) or (N,) predicted probabilities
        n_bins: 구간 수

    Returns:
        ECE 스칼라 값
    """
    y_true = y_true.flatten()
    y_prob = y_prob.flatten()

    bins     = np.linspace(0.0, 1.0, n_bins + 1)
    ece      = 0.0
    n_total  = len(y_true)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n_total) * abs(acc - conf)

    return float(ece)


def compute_mce(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Maximum Calibration Error (MCE) 계산.

    모든 구간 중 정확도와 신뢰도의 가장 큰 차이를 반환합니다.
    보고서 표 예시: MCE Before Scaling=0.1234, After=0.0678

    Args:
        y_true: (N, C) or (N,) binary labels
        y_prob: (N, C) or (N,) predicted probabilities
        n_bins: 구간 수

    Returns:
        MCE 스칼라 값
    """
    y_true = y_true.flatten()
    y_prob = y_prob.flatten()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mce  = 0.0

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        mce  = max(mce, abs(acc - conf))

    return float(mce)


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    ECE 및 MCE를 한 번에 계산하는 헬퍼 함수.

    README Calibration 표 재현:
        | Metric | Before Scaling | After Temp Scaling |
        | ECE    |     0.0823     |       0.0456       |
        | MCE    |     0.1234     |       0.0678       |

    Returns:
        {"ece": float, "mce": float}
    """
    return {
        "ece": compute_ece(y_true, y_prob, n_bins),
        "mce": compute_mce(y_true, y_prob, n_bins),
    }


class TemperatureScaling(nn.Module):
    """
    Temperature Scaling for multi-label calibration.

    단일 스칼라 파라미터 T로 logit을 나누어 calibration 개선.

    Usage:
        ts = TemperatureScaling()
        ts.fit(logits_val, labels_val)
        calibrated_probs = ts(logits_test)
    """

    def __init__(self):
        assert _TORCH_AVAILABLE, "pip install torch 필요"
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: "torch.Tensor") -> "torch.Tensor":
        import torch.nn.functional as F
        return torch.sigmoid(logits / self.temperature)

    def fit(
        self,
        logits: "torch.Tensor",
        labels: "torch.Tensor",
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """NLL 최소화로 Temperature 파라미터 최적화."""
        import torch.nn.functional as F
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                logits / self.temperature, labels
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.item())
