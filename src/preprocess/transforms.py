"""
Image Preprocessing & Augmentation Transforms for CXR-CAD.

cv2.createCLAHE 기반 실제 CLAHE 구현 포함.
학습용(augmentation) 및 추론용(deterministic) 파이프라인 제공.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms


# ── 상수 ─────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── CLAHE 구현 ─────────────────────────────────────────────────────────────────

def apply_clahe(
    image: Image.Image,
    clip_limit: float = 4.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> Image.Image:
    """
    cv2.createCLAHE를 사용한 실제 CLAHE 적용.

    PIL 이미지 → Grayscale → cv2 CLAHE → RGB 변환.

    Args:
        image: 입력 PIL Image
        clip_limit: CLAHE 클립 한계 (높을수록 강한 대비 강조)
        tile_grid_size: CLAHE 타일 크기

    Returns:
        CLAHE 적용된 RGB PIL Image
    """
    # PIL → numpy (Grayscale)
    gray = np.array(image.convert("L"))

    # cv2 CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)

    # numpy → PIL RGB
    return Image.fromarray(enhanced, mode="L").convert("RGB")


class CLAHETransform:
    """torchvision Compose 호환 CLAHE 변환 클래스."""

    def __init__(
        self,
        clip_limit: float = 4.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        return apply_clahe(img, self.clip_limit, self.tile_grid_size)

    def __repr__(self) -> str:
        return (
            f"CLAHETransform(clip_limit={self.clip_limit}, "
            f"tile_grid_size={self.tile_grid_size})"
        )


# ── Transform Pipelines ──────────────────────────────────────────────────────

def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    학습용 augmentation 파이프라인.

    CLAHE → Resize → RandomHFlip → RandomRotation(±10°) →
    RandomAffine → ColorJitter → ToTensor → Normalize
    """
    return transforms.Compose([
        CLAHETransform(clip_limit=4.0, tile_grid_size=(8, 8)),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms(image_size: int = 224) -> transforms.Compose:
    """
    추론용 결정론적 파이프라인 (augmentation 없음).

    CLAHE → Resize → ToTensor → Normalize
    """
    return transforms.Compose([
        CLAHETransform(clip_limit=4.0, tile_grid_size=(8, 8)),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_tta_transforms(image_size: int = 224) -> list:
    """
    Test-Time Augmentation용 변환 목록.

    원본 + 좌우 반전 + 회전 5° + 회전 -5° 총 4가지 변환.
    train.py의 TTA 래퍼에서 사용.

    Returns:
        List[transforms.Compose]
    """
    base = [
        CLAHETransform(clip_limit=4.0, tile_grid_size=(8, 8)),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    tta_list = [
        transforms.Compose(base),  # 원본
        transforms.Compose([
            CLAHETransform(),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        transforms.Compose([
            CLAHETransform(),
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=(5, 5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        transforms.Compose([
            CLAHETransform(),
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=(-5, -5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
    ]
    return tta_list


def preprocess_single_image(
    image: Image.Image,
    image_size: int = 224,
) -> torch.Tensor:
    """
    단일 PIL Image를 모델 입력 텐서로 변환.

    Returns:
        Tensor of shape (1, 3, 224, 224)
    """
    transform = get_inference_transforms(image_size)
    tensor = transform(image)
    return tensor.unsqueeze(0)
