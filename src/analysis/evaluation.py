"""
모델 평가 지표 계산.

- AUROC (per-class & macro-average)
- AUPRC (per-class & macro-average)
- Operating Point 분석
    · Youden's J  : Sensitivity + Specificity 최대화
    · Sens@Spec90 : Specificity >= 0.90 제약하에 최대 Sensitivity (확진 보조)
    · Spec@Sens90 : Sensitivity >= 0.90 제약하에 최대 Specificity (스크리닝)
  각 기준별 Threshold / Sensitivity / Specificity / PPV / NPV 반환
- MCE (Maximum Calibration Error) 보조 함수
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        roc_curve,
    )
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

from src.train.models import DISEASE_LABELS


def compute_auroc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: List[str] = DISEASE_LABELS,
) -> Dict[str, float]:
    """
    클래스별 및 macro-average AUROC 계산.

    Args:
        y_true: (N, C) binary labels
        y_prob: (N, C) predicted probabilities
        labels: 클래스 이름 리스트

    Returns:
        {"Atelectasis": 0.82, ..., "macro_avg": 0.79}
    """
    assert _SKLEARN_AVAILABLE, "pip install scikit-learn 필요"

    result = {}
    for i, label in enumerate(labels):
        try:
            result[label] = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            result[label] = float("nan")

    valid = [v for v in result.values() if not np.isnan(v)]
    result["macro_avg"] = float(np.mean(valid)) if valid else float("nan")
    return result


def compute_auprc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: List[str] = DISEASE_LABELS,
) -> Dict[str, float]:
    """
    클래스별 및 macro-average AUPRC 계산.
    """
    assert _SKLEARN_AVAILABLE, "pip install scikit-learn 필요"

    result = {}
    for i, label in enumerate(labels):
        try:
            result[label] = average_precision_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            result[label] = float("nan")

    valid = [v for v in result.values() if not np.isnan(v)]
    result["macro_avg"] = float(np.mean(valid)) if valid else float("nan")
    return result


def find_operating_points(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: List[str] = DISEASE_LABELS,
) -> Dict[str, float]:
    """
    각 클래스에 대해 Youden's J index 기준 최적 임계값 탐색.

    Returns:
        {"Atelectasis": 0.31, "Cardiomegaly": 0.42, ...}
    """
    assert _SKLEARN_AVAILABLE, "pip install scikit-learn 필요"

    thresholds = {}
    for i, label in enumerate(labels):
        try:
            fpr, tpr, thresh = roc_curve(y_true[:, i], y_prob[:, i])
            j_scores = tpr - fpr
            best_idx = int(np.argmax(j_scores))
            thresholds[label] = float(thresh[best_idx])
        except Exception:
            thresholds[label] = 0.5
    return thresholds


def _operating_point_metrics(
    y_true_bin: np.ndarray,
    y_prob_bin: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """단일 임계값에서 Sensitivity / Specificity / PPV / NPV 계산."""
    pred = (y_prob_bin >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true_bin == 1)).sum())
    fp = int(((pred == 1) & (y_true_bin == 0)).sum())
    tn = int(((pred == 0) & (y_true_bin == 0)).sum())
    fn = int(((pred == 0) & (y_true_bin == 1)).sum())

    sensitivity  = tp / max(tp + fn, 1)
    specificity  = tn / max(tn + fp, 1)
    ppv          = tp / max(tp + fp, 1)   # Precision / Positive Predictive Value
    npv          = tn / max(tn + fn, 1)   # Negative Predictive Value
    return {
        "threshold":   round(threshold, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "ppv":         round(ppv, 4),
        "npv":         round(npv, 4),
    }


def find_operating_points_detail(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str,
    label_idx: Optional[int] = None,
    target_spec: float = 0.90,
    target_sens: float = 0.90,
) -> Dict[str, Dict[str, float]]:
    """
    하나의 질환에 대해 세 기준 Operating Point를 모두 계산.

    README 표 재현:
        | 기준         | Threshold | Sensitivity | Specificity | PPV   | NPV   |
        | Youden's J   |  0.42     |  0.823      |  0.845      | 0.134 | 0.992 |
        | Sens@Spec90  |  0.28     |  0.900      |  0.712      | 0.081 | 0.996 |
        | Spec@Sens90  |  0.56     |  0.689      |  0.900      | 0.165 | 0.988 |

    Args:
        y_true     : (N, C) 또는 (N,) binary labels
        y_prob     : (N, C) 또는 (N,) 예측 확률
        label      : 질환 이름 (로그용)
        label_idx  : y_true/y_prob이 (N, C)일 때 열 인덱스
        target_spec: Sens@Spec 기준 목표 Specificity (default 0.90)
        target_sens: Spec@Sens 기준 목표 Sensitivity (default 0.90)

    Returns:
        {
            "youden"    : {threshold, sensitivity, specificity, ppv, npv},
            "sens_spec90": {threshold, sensitivity, specificity, ppv, npv},
            "spec_sens90": {threshold, sensitivity, specificity, ppv, npv},
        }
    """
    assert _SKLEARN_AVAILABLE, "pip install scikit-learn 필요"

    if label_idx is not None:
        yt = y_true[:, label_idx].astype(int)
        yp = y_prob[:, label_idx]
    else:
        yt = y_true.flatten().astype(int)
        yp = y_prob.flatten()

    try:
        fpr, tpr, thresh_arr = roc_curve(yt, yp)
    except Exception as e:
        raise ValueError(f"[{label}] roc_curve 실패: {e}")

    spec_arr = 1.0 - fpr   # Specificity = 1 - FPR

    # ── 1. Youden's J ──────────────────────────────────────────────────────
    j_idx        = int(np.argmax(tpr - fpr))
    thresh_youden = float(thresh_arr[j_idx])

    # ── 2. Sens@Spec90 : Specificity >= target_spec 유지하며 Sensitivity 최대 ─
    spec90_mask   = spec_arr >= target_spec
    if spec90_mask.any():
        best_idx_s    = int(np.argmax(tpr * spec90_mask))
        thresh_s_sp90 = float(thresh_arr[best_idx_s])
    else:
        thresh_s_sp90 = float(thresh_arr[-1])

    # ── 3. Spec@Sens90 : Sensitivity >= target_sens 유지하며 Specificity 최대 ─
    sens90_mask   = tpr >= target_sens
    if sens90_mask.any():
        best_idx_sp   = int(np.argmax(spec_arr * sens90_mask))
        thresh_sp_s90 = float(thresh_arr[best_idx_sp])
    else:
        thresh_sp_s90 = float(thresh_arr[0])

    return {
        "youden":      _operating_point_metrics(yt, yp, thresh_youden),
        "sens_spec90": _operating_point_metrics(yt, yp, thresh_s_sp90),
        "spec_sens90": _operating_point_metrics(yt, yp, thresh_sp_s90),
    }
