"""
Multi-hot Encoding 테스트.

NIH ChestX-ray14 레이블 → 멀티-핫 벡터 변환 검증.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.train.models import DISEASE_LABELS, NUM_CLASSES


def _make_multihot(labels: list[str]) -> torch.Tensor:
    """질환 이름 리스트 → 14차원 멀티-핫 텐서."""
    vec = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for label in labels:
        if label in DISEASE_LABELS:
            vec[DISEASE_LABELS.index(label)] = 1.0
    return vec


def test_multihot_no_finding():
    vec = _make_multihot([])
    assert vec.sum().item() == 0.0


def test_multihot_single_disease():
    vec = _make_multihot(["Cardiomegaly"])
    assert vec.shape == (NUM_CLASSES,)
    assert vec[DISEASE_LABELS.index("Cardiomegaly")] == 1.0
    assert vec.sum().item() == 1.0


def test_multihot_multiple():
    targets = ["Atelectasis", "Effusion", "Pneumonia"]
    vec = _make_multihot(targets)
    for t in targets:
        assert vec[DISEASE_LABELS.index(t)] == 1.0
    assert vec.sum().item() == len(targets)


def test_multihot_all_diseases():
    vec = _make_multihot(DISEASE_LABELS)
    assert vec.sum().item() == NUM_CLASSES


def test_disease_labels_count():
    assert NUM_CLASSES == 14
    assert len(DISEASE_LABELS) == 14
