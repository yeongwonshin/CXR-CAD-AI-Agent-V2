"""
NIH ChestX-ray14 Data Loader.

실제 데이터 로딩, Patient ID 기준 5-Fold GroupKFold Split,
데이터 누수 검증, pos_weight 계산을 제공합니다.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from glob import glob # glob 추가

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset, DataLoader

# ── 상수 ─────────────────────────────────────────────────────────────────────

DISEASE_LABELS: List[str] = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

NUM_CLASSES = len(DISEASE_LABELS)

_COL_IMAGE    = "Image Index"
_COL_FINDING  = "Finding Labels"
_COL_PATIENT  = "Patient ID"
_COL_AGE      = "Patient Age"
_COL_SEX      = "Patient Gender"
_COL_VIEW     = "View Position"
_COL_FOLLOW   = "Follow-up #"


# ── CSV 파싱 ──────────────────────────────────────────────────────────────────

def load_nih_csv(data_root: str) -> pd.DataFrame:
    """
    NIH CSV를 읽어 멀티-핫 레이블 컬럼을 추가한 DataFrame 반환.
    """
    data_root_path = Path(data_root)
    csv_candidates = [
        data_root_path / "Data_Entry_2017.csv",
        data_root_path / "Data_Entry_2017_v2020.csv",
    ]
    csv_candidates.extend(sorted(data_root_path.glob("Data_Entry_2017*.csv")))
    csv_path = next((path for path in csv_candidates if path.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            f"NIH metadata CSV not found under {data_root}. "
            "Expected Data_Entry_2017.csv or Data_Entry_2017*.csv."
        )
    
    df = pd.read_csv(csv_path)
    
    # 캐글 서버 구조(images_001/images/...) 혹은 하위 폴더 어디든 이미지가 있는 경우 대응
    all_image_paths = glob(os.path.join(data_root, "**", "*.png"), recursive=True)
    
    image_path_dict = {os.path.basename(x): x for x in all_image_paths}
    df['Full_Path'] = df[_COL_IMAGE].map(image_path_dict)

    for disease in DISEASE_LABELS:
        df[disease] = df[_COL_FINDING].apply(
            lambda findings: 1.0 if disease in findings.split("|") else 0.0
        )

    return df


def compute_pos_weight(df: pd.DataFrame) -> torch.Tensor:
    pos_counts = df[DISEASE_LABELS].sum(axis=0).values.astype(float)
    neg_counts = len(df) - pos_counts
    pos_weight = neg_counts / np.clip(pos_counts, 1, None)
    return torch.tensor(pos_weight, dtype=torch.float32)


# ── Train / Val / Test Split ──────────────────────────────────────────────────

def split_by_patient(
    df: pd.DataFrame,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    patients = df[_COL_PATIENT].unique()
    rng.shuffle(patients)

    n_test = max(1, int(len(patients) * test_ratio))

    test_patients  = set(patients[:n_test])
    train_patients = set(patients[n_test:])

    train_df = df[df[_COL_PATIENT].isin(train_patients)].reset_index(drop=True)
    test_df  = df[df[_COL_PATIENT].isin(test_patients)].reset_index(drop=True)

    return train_df, test_df


def verify_no_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> bool:
    train_pts = set(train_df[_COL_PATIENT])
    test_pts  = set(test_df[_COL_PATIENT])

    assert train_pts.isdisjoint(test_pts), "❌ 데이터 누수: Train ∩ Test 환자 존재"
    print(
        f"✅ 데이터 누수 없음 확인\n"
        f"   Train: {len(train_df):,} images / {len(train_pts):,} patients\n"
        f"   Test : {len(test_df):,} images / {len(test_pts):,} patients"
    )
    return True


def get_group_kfold_splits(
    train_df: pd.DataFrame,
    n_splits: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    gkf = GroupKFold(n_splits=n_splits)
    groups = train_df[_COL_PATIENT].values
    X = np.arange(len(train_df))
    splits = list(gkf.split(X, groups=groups))
    print(f"✅ {n_splits}-Fold GroupKFold Splits 준비 완료")
    return splits


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class NIHChestXrayDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str,
        transform=None,
        return_meta: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.return_meta = return_meta

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        
        if 'Full_Path' in row and pd.notna(row['Full_Path']):
            img_path = Path(row['Full_Path'])
        else:
            img_path = self.images_dir / row[_COL_IMAGE]

        if img_path.exists():
            image = Image.open(img_path).convert("RGB")
        else:
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        if self.transform is not None:
            image = self.transform(image)
        else:
            import torchvision.transforms as T
            image = T.ToTensor()(image)

        label = torch.tensor(
            row[DISEASE_LABELS].values.astype(np.float32),
            dtype=torch.float32,
        )

        if self.return_meta:
            meta = {
                "image_index":    row[_COL_IMAGE],
                "patient_id":     row[_COL_PATIENT],
                "patient_age":    row.get(_COL_AGE, -1),
                "patient_sex":    row.get(_COL_SEX, "Unknown"),
                "view_position":  row.get(_COL_VIEW, "Unknown"),
            }
            return image, label, meta

        return image, label


# ── DataLoader Factory ────────────────────────────────────────────────────────

def create_dataloader(
    df: pd.DataFrame,
    images_dir: str,
    transform=None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = False,
    return_meta: bool = False,
) -> DataLoader:
    dataset = NIHChestXrayDataset(
        df=df,
        images_dir=images_dir,
        transform=transform,
        return_meta=return_meta,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )


# ── 고수준 팩토리 ─────────────────────────────────────────────────────────────

def build_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform=None,
    eval_transform=None,
    test_ratio: float = 0.15,
) -> Dict[str, DataLoader]:
    images_dir = os.path.join(data_root, "images")

    df = load_nih_csv(data_root)
    train_df, test_df = split_by_patient(df, test_ratio)
    verify_no_leakage(train_df, test_df)

    pos_weight = compute_pos_weight(train_df)

    # val 관련 코드 완전 삭제 (5-Fold에 맞춰 수정)
    return {
        "train":      create_dataloader(train_df, images_dir, train_transform, batch_size, num_workers, shuffle=True),
        "test":       create_dataloader(test_df,  images_dir, eval_transform,  batch_size, num_workers, shuffle=False),
        "pos_weight": pos_weight,
        "train_df":   train_df,
        "test_df":    test_df,
    }
