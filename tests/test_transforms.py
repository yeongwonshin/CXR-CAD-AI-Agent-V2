"""
전처리 테스트.

CLAHE 변환, get_train_transforms, get_inference_transforms 동작 검증.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.preprocess.transforms import (
    CLAHETransform,
    apply_clahe,
    get_inference_transforms,
    get_train_transforms,
    preprocess_single_image,
)


def _make_gray_pil(size: int = 64) -> Image.Image:
    arr = (np.random.rand(size, size) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def test_apply_clahe_returns_rgb():
    img    = _make_gray_pil()
    result = apply_clahe(img)
    assert result.mode == "RGB"
    assert result.size == img.size


def test_clahe_transform_composable():
    transform = CLAHETransform()
    img    = _make_gray_pil()
    result = transform(img)
    assert isinstance(result, Image.Image)


def test_train_transforms_output_shape():
    transform = get_train_transforms(image_size=224)
    img    = _make_gray_pil(256)
    tensor = transform(img)
    assert tensor.shape == (3, 224, 224)


def test_inference_transforms_deterministic():
    transform = get_inference_transforms(image_size=224)
    img = _make_gray_pil(256)
    t1  = transform(img)
    t2  = transform(img)
    import torch
    assert torch.allclose(t1, t2), "추론 변환은 결정론적이어야 합니다"


def test_preprocess_single_image_shape():
    img    = _make_gray_pil(256)
    tensor = preprocess_single_image(img, image_size=224)
    assert tensor.shape == (1, 3, 224, 224)
