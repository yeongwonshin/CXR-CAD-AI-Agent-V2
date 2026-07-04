"""
Subgroup Analysis 유틸리티.

성별(M/F), 연령대, 촬영구도(View Position: PA/AP)별 AUROC 성능 분석.
데이터 편향 및 공정성 문제 탐지에 활용.

사용 예시:
    - group_col="Patient Gender"        → 성별 분석
    - group_col="Age Group"              → 연령대 분석
    - group_col="View Position"          → PA / AP 촬영구도 분석
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.analysis.evaluation import compute_auroc
from src.train.models import DISEASE_LABELS


def subgroup_auroc(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    group_col: str,
    labels: List[str] = DISEASE_LABELS,
) -> Dict[str, Dict[str, float]]:
    """
    지정 컬럼 기준 그룹별 AUROC 계산.

    Args:
        df       : 메타데이터 DataFrame (인덱스 정렬 필요)
        y_prob   : (N, C) 예측 확률
        group_col: 그룹 컬럼명
                   예) "Patient Gender" (M/F)
                       "Age Group"      (0-20, 20-40, ...)
                       "View Position"  (PA, AP)
        labels   : 평가 레이블 리스트

    Returns:
        {그룹값: {레이블: AUROC, "macro_avg": AUROC}}
    """
    results = {}
    for group_val, group_df in df.groupby(group_col):
        idx    = group_df.index
        y_true = group_df[labels].values.astype(np.float32)
        y_pred = y_prob[idx]
        results[str(group_val)] = compute_auroc(y_true, y_pred, labels)
    return results


def age_group_auroc(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    age_col: str = "Patient Age",
    bins: List[int] = [0, 20, 40, 60, 80, 120],
    labels: List[str] = DISEASE_LABELS,
) -> Dict[str, Dict[str, float]]:
    """
    연령대 구간별 AUROC 계산.
    """
    age_labels = [f"{lo}-{hi}" for lo, hi in zip(bins[:-1], bins[1:])]
    df = df.copy()
    df["__age_group__"] = pd.cut(
        df[age_col], bins=bins, labels=age_labels, right=False
    )
    return subgroup_auroc(df, y_prob, "__age_group__", labels)
