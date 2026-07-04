"""
PyTorch Dataset 뼈대.

NIH ChestX-ray14 멀티-레이블 분류용 Dataset 클래스.
실제 구현은 노트북 01_EDA.ipynb 및 04_Training.ipynb 참고.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.train.models import DISEASE_LABELS


class ChestXrayDataset(Dataset):
    """
    NIH ChestX-ray14 멀티-레이블 Dataset.

    Args:
        df        : 메타데이터 DataFrame. 최소 컬럼: ['Image Index', *DISEASE_LABELS]
        image_dir : 이미지 파일이 위치한 디렉토리 경로
        transform : torchvision transforms (train/inference 분리)
        labels    : 예측할 질환 레이블 리스트 (기본: DISEASE_LABELS)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str | Path,
        transform: Optional[Callable] = None,
        labels: List[str] = DISEASE_LABELS,
    ):
        self.df        = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.labels    = labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row      = self.df.iloc[idx]
        img_path = self.image_dir / row["Image Index"]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(
            [float(row[lbl]) for lbl in self.labels], dtype=torch.float32
        )
        return image, label
