"""
External Validation 유틸리티.

내부 NIH 테스트셋 성능과 외부 데이터셋(CheXpert, PadChest 등)
성능을 비교하여 일반화 능력을 검증합니다.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.analysis.evaluation import compute_auroc, compute_auprc
from src.train.models import DISEASE_LABELS


def evaluate_external(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    dataset_name: str,
    labels: List[str] = DISEASE_LABELS,
) -> Dict[str, Dict]:
    """
    외부 데이터셋에 대한 AUROC/AUPRC 계산.

    Args:
        y_true      : (N, C) binary labels
        y_prob      : (N, C) predicted probabilities
        dataset_name: 데이터셋 이름 (결과 딕셔너리 키)
        labels      : 평가 레이블

    Returns:
        {dataset_name: {"auroc": {...}, "auprc": {...}}}
    """
    return {
        dataset_name: {
            "auroc": compute_auroc(y_true, y_prob, labels),
            "auprc": compute_auprc(y_true, y_prob, labels),
        }
    }


def compare_internal_external(
    internal: Dict[str, float],
    external: Dict[str, float],
    labels: List[str] = DISEASE_LABELS,
) -> Dict[str, float]:
    """
    내부/외부 AUROC 차이(gap) 계산.

    Args:
        internal: 내부 테스트셋 AUROC dict
        external: 외부 데이터셋 AUROC dict

    Returns:
        {"Atelectasis": -0.03, ..., "macro_avg": -0.05}  (외부 - 내부)
    """
    return {
        label: external.get(label, float("nan")) - internal.get(label, float("nan"))
        for label in [*labels, "macro_avg"]
    }
