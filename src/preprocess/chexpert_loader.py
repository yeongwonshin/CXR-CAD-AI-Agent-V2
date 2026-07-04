"""
CheXpert Dataset Loader.

Stanford CheXpert-v1.0-small 기준 구현.
NIH 14-class 모델로 External Validation할 때 사용합니다.

데이터셋 구조 (Stanford 라이선스 신청 후 다운로드):
    CheXpert-v1.0-small/
    ├── train/
    │   ├── patient00001/study1/view1_frontal.jpg
    │   └── ...
    ├── valid/
    │   ├── patient00001/study1/view1_frontal.jpg
    │   └── ...
    ├── train.csv
    └── valid.csv

CSV 컬럼:
    Path, Sex, Age, Frontal/Lateral, AP/PA,
    No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity,
    Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis,
    Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices

레이블 값:
    1.0  = 양성 (Positive)
    0.0  = 음성 (Negative)
   -1.0  = 불확실 (Uncertain) → uncertain_strategy에 따라 처리
    NaN  = 언급 없음 → 0.0 처리

NIH ↔ CheXpert 공통 7개 클래스:
    Atelectasis, Cardiomegaly, Consolidation, Edema,
    Effusion (← Pleural Effusion), Pneumonia, Pneumothorax
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.train.models import DISEASE_LABELS

# ── 레이블 매핑 ───────────────────────────────────────────────────────────────

# CheXpert CSV 컬럼명 → NIH DISEASE_LABELS 내 이름
# (NIH에서 사용하는 이름을 기준으로 매핑)
CHEXPERT_TO_NIH: Dict[str, str] = {
    "Atelectasis":    "Atelectasis",
    "Cardiomegaly":   "Cardiomegaly",
    "Consolidation":  "Consolidation",
    "Edema":          "Edema",
    "Pleural Effusion": "Effusion",   # CheXpert 이름이 다름
    "Pneumonia":      "Pneumonia",
    "Pneumothorax":   "Pneumothorax",
}

# External Validation에 사용할 NIH 레이블 (7개 공통 클래스)
EVAL_LABELS: List[str] = list(CHEXPERT_TO_NIH.values())

# CheXpert 전체 라벨 컬럼
CHEXPERT_ALL_LABELS: List[str] = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]

# External Validation에 사용할 CheXpert 컬럼 (CHEXPERT_TO_NIH의 key들)
CHEXPERT_EVAL_COLS: List[str] = list(CHEXPERT_TO_NIH.keys())

UncertainStrategy = Literal["u_zeros", "u_ones", "u_ignore"]


# ── CSV 파싱 ──────────────────────────────────────────────────────────────────

def load_chexpert_csv(
    csv_path: str,
    chexpert_root: str,
    uncertain_strategy: UncertainStrategy = "u_zeros",
    frontal_only: bool = True,
) -> pd.DataFrame:
    """
    CheXpert CSV를 읽어 External Validation용 DataFrame 반환.

    Args:
        csv_path           : train.csv 또는 valid.csv 경로
        chexpert_root      : CheXpert-v1.0-small 루트 (abs_path 계산용)
        uncertain_strategy : 불확실(-1) 처리 방법
            - "u_zeros" : -1 → 0 (보수적, 논문 기본)
            - "u_ones"  : -1 → 1 (낙관적)
            - "u_ignore": -1 포함 샘플 제외 (샘플 수 감소)
        frontal_only       : True이면 Frontal/Lateral==Frontal 만 사용 (권장)

    Returns:
        DataFrame with columns: abs_path + 7개 NIH 레이블명
    """
    df = pd.read_csv(csv_path)

    # Frontal 이미지만 선택
    if frontal_only and "Frontal/Lateral" in df.columns:
        df = df[df["Frontal/Lateral"] == "Frontal"].copy()

    # NaN → 0.0 (언급 없음 = 음성)
    for col in CHEXPERT_ALL_LABELS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Uncertain 처리
    if uncertain_strategy == "u_zeros":
        for col in CHEXPERT_ALL_LABELS:
            if col in df.columns:
                df[col] = df[col].replace(-1.0, 0.0)
    elif uncertain_strategy == "u_ones":
        for col in CHEXPERT_ALL_LABELS:
            if col in df.columns:
                df[col] = df[col].replace(-1.0, 1.0)
    elif uncertain_strategy == "u_ignore":
        mask = (df[CHEXPERT_EVAL_COLS] != -1.0).all(axis=1)
        df = df[mask].copy()
        if len(df) == 0:
            raise ValueError("u_ignore 전략으로 모든 샘플이 제거되었습니다.")

    # CheXpert 레이블 → NIH 레이블명으로 리네임
    rename_map = {chex_col: nih_col for chex_col, nih_col in CHEXPERT_TO_NIH.items()}
    df = df.rename(columns=rename_map)

    # 이미지 절대 경로 계산
    # CheXpert CSV의 Path 컬럼 예: "CheXpert-v1.0-small/valid/patient00001/..."
    root = Path(chexpert_root)
    if "Path" in df.columns:
        # Path 컬럼이 "CheXpert-v1.0-small/..." 형태인 경우 처리
        df["abs_path"] = df["Path"].apply(
            lambda p: str(root / Path(p).relative_to(Path(p).parts[0]))
            if Path(p).parts[0] in ("CheXpert-v1.0-small", "CheXpert-v1.0")
            else str(root / p)
        )
    else:
        raise ValueError("CSV에 'Path' 컬럼이 없습니다.")

    df = df.reset_index(drop=True)
    print(
        f"[CheXpert] {Path(csv_path).name} 로드 완료\n"
        f"  총 샘플: {len(df):,} (uncertain_strategy={uncertain_strategy}, "
        f"frontal_only={frontal_only})"
    )
    return df


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class CheXpertDataset(Dataset):
    """
    CheXpert External Validation Dataset.

    NIH 모델로 추론하기 위해 NIH와 동일한 전처리 파이프라인 사용.
    레이블은 EVAL_LABELS(7개)만 반환하며, NIH DISEASE_LABELS 순서에 맞추기
    위해 14-dim 벡터로 패딩 가능 (pad_to_nih=True).

    Args:
        df            : load_chexpert_csv() 반환 DataFrame
        transform     : torchvision 변환 파이프라인
        pad_to_nih    : True이면 (B, 14) 반환 (NIH 모델과 동일한 출력 차원)
                        False이면 (B, 7) 반환
        return_path   : True이면 (image, label, abs_path) 반환
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        pad_to_nih: bool = True,
        return_path: bool = False,
    ):
        self.df         = df.reset_index(drop=True)
        self.transform  = transform
        self.pad_to_nih = pad_to_nih
        self.return_path = return_path

        # EVAL_LABELS이 df에 있는지 확인
        missing = [c for c in EVAL_LABELS if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame에 다음 NIH 레이블 컬럼이 없습니다: {missing}")

        # NIH 14-class 순서에서 eval_labels의 인덱스 (패딩용)
        self._eval_indices = [DISEASE_LABELS.index(l) for l in EVAL_LABELS]

        # 이미지 파일 존재 여부 경고
        missing_files = [
            p for p in self.df["abs_path"].head(5).tolist()
            if not Path(p).exists()
        ]
        if missing_files:
            warnings.warn(
                f"⚠️  CheXpert 이미지 없음 (예시): {missing_files[:2]}\n"
                f"   chexpert_root 경로가 올바른지 확인하세요.",
                UserWarning,
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # 이미지 로드
        img_path = Path(row["abs_path"])
        if img_path.exists():
            image = Image.open(img_path).convert("RGB")
        else:
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        if self.transform is not None:
            image = self.transform(image)
        else:
            import torchvision.transforms as T
            image = T.ToTensor()(image)

        # 7개 레이블 추출
        label_7 = torch.tensor(
            row[EVAL_LABELS].values.astype(np.float32),
            dtype=torch.float32,
        )

        if self.pad_to_nih:
            # 14-dim 벡터로 패딩 (NIH 모델 출력과 동일한 차원)
            label_14 = torch.zeros(len(DISEASE_LABELS), dtype=torch.float32)
            for i, nih_idx in enumerate(self._eval_indices):
                label_14[nih_idx] = label_7[i]
            label = label_14
        else:
            label = label_7

        if self.return_path:
            return image, label, str(row["abs_path"])
        return image, label


# ── DataLoader Factory ────────────────────────────────────────────────────────

def create_chexpert_dataloader(
    df: pd.DataFrame,
    transform=None,
    batch_size: int = 64,
    num_workers: int = 4,
    pad_to_nih: bool = True,
) -> DataLoader:
    """
    CheXpertDataset → DataLoader 팩토리.

    Args:
        df         : load_chexpert_csv() 반환 DataFrame
        transform  : 전처리 파이프라인 (get_inference_transforms() 권장)
        batch_size : 배치 크기
        num_workers: 병렬 로드 수
        pad_to_nih : True이면 (B, 14) 레이블 반환

    Returns:
        DataLoader
    """
    dataset = CheXpertDataset(df=df, transform=transform, pad_to_nih=pad_to_nih)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


# ── 고수준 헬퍼 ───────────────────────────────────────────────────────────────

def build_chexpert_val_loader(
    chexpert_root: str,
    split: str = "valid",
    uncertain_strategy: UncertainStrategy = "u_zeros",
    transform=None,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, List[str]]:
    """
    CheXpert External Validation DataLoader 원스톱 생성.

    Args:
        chexpert_root      : CheXpert-v1.0-small 루트 디렉토리
        split              : "valid" (권장, 5200장) | "train"
        uncertain_strategy : Uncertain 처리 전략
        transform          : 전처리 파이프라인
        batch_size         : 배치 크기
        num_workers        : 병렬 로드 수

    Returns:
        (DataLoader, eval_labels)  ← eval_labels = 7개 NIH 레이블 이름
    """
    root = Path(chexpert_root)
    csv_path = str(root / f"{split}.csv")

    df = load_chexpert_csv(
        csv_path=csv_path,
        chexpert_root=chexpert_root,
        uncertain_strategy=uncertain_strategy,
        frontal_only=True,
    )
    loader = create_chexpert_dataloader(
        df=df,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        pad_to_nih=True,
    )
    return loader, EVAL_LABELS
