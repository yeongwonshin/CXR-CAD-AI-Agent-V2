"""
Patient-wise Train/Validation/Test Split 유틸리티.

NIH ChestX-ray14는 동일 환자의 이미지가 여러 장 포함될 수 있으므로,
환자 ID 기준으로 분할하여 data leakage를 방지합니다.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def patient_wise_split(
    df: pd.DataFrame,
    patient_col: str = "Patient ID",
    val_ratio: float = 0.0,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    환자 ID 기준으로 Train / Val / Test 분할.

    Args:
        df         : 전체 메타데이터 DataFrame
        patient_col: 환자 ID 컬럼명 (NIH: "Patient ID")
        val_ratio  : Validation 비율
        test_ratio : Test 비율
        seed       : 재현성을 위한 random seed

    Returns:
        (train_df, val_df, test_df)
    """
    rng        = np.random.default_rng(seed)
    patients   = df[patient_col].unique()
    rng.shuffle(patients)

    n          = len(patients)
    n_test     = int(n * test_ratio)
    n_val      = int(n * val_ratio)

    test_pats  = set(patients[:n_test])
    val_pats   = set(patients[n_test:n_test + n_val])
    train_pats = set(patients[n_test + n_val:])

    train_df = df[df[patient_col].isin(train_pats)].reset_index(drop=True)
    val_df   = df[df[patient_col].isin(val_pats)].reset_index(drop=True)
    test_df  = df[df[patient_col].isin(test_pats)].reset_index(drop=True)

    print(
        f"Split (patient-wise) — "
        f"train: {len(train_df)} imgs ({len(train_pats)} pts), "
        f"val: {len(val_df)} imgs ({len(val_pats)} pts), "
        f"test: {len(test_df)} imgs ({len(test_pats)} pts)"
    )
    return train_df, val_df, test_df
