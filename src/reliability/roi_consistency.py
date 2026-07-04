"""ROI-based explanation consistency checks for chest X-ray models.

The paper already uses Grad-CAM error review, but Grad-CAM alone can be
subjective. This module turns a heatmap and a lung/lesion ROI mask into
quantitative deployment-readiness scores:

* roi_energy_ratio: fraction of explanation energy inside the ROI
* outside_energy_ratio: fraction outside the ROI
* shortcut_flag: True when too much energy is outside the ROI

It is intentionally model-agnostic and can consume Grad-CAM, Eigen-CAM,
attention rollout, or any non-negative explanation map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass(frozen=True)
class ROIConsistencyResult:
    roi_energy_ratio: float
    outside_energy_ratio: float
    shortcut_flag: bool
    roi_pixels: int
    total_energy: float


def _as_float_array(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape={arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def compute_roi_consistency(
    heatmap: np.ndarray,
    roi_mask: np.ndarray,
    outside_threshold: float = 0.40,
    eps: float = 1e-8,
) -> ROIConsistencyResult:
    """Compute how much explanation energy lies inside the valid ROI.

    Args:
        heatmap: 2D explanation map. Negative values are clipped to zero.
        roi_mask: 2D binary or soft ROI mask. Values > 0 are treated as ROI.
        outside_threshold: shortcut flag is raised when outside energy exceeds
            this ratio.
        eps: numerical stability constant.
    """
    h = _as_float_array(heatmap, "heatmap")
    m = _as_float_array(roi_mask, "roi_mask")
    if h.shape != m.shape:
        raise ValueError(f"heatmap and roi_mask shapes differ: {h.shape} vs {m.shape}")
    if not 0.0 <= outside_threshold <= 1.0:
        raise ValueError("outside_threshold must be in [0, 1]")

    h = np.clip(h, 0.0, None)
    roi = m > 0
    total = float(h.sum())
    if total <= eps:
        return ROIConsistencyResult(0.0, 1.0, True, int(roi.sum()), 0.0)

    inside = float(h[roi].sum())
    roi_ratio = inside / total
    outside_ratio = 1.0 - roi_ratio
    return ROIConsistencyResult(
        roi_energy_ratio=round(float(roi_ratio), 6),
        outside_energy_ratio=round(float(outside_ratio), 6),
        shortcut_flag=bool(outside_ratio >= outside_threshold),
        roi_pixels=int(roi.sum()),
        total_energy=round(total, 6),
    )


def batch_roi_consistency(
    heatmaps: Iterable[np.ndarray],
    roi_masks: Iterable[np.ndarray],
    outside_threshold: float = 0.40,
) -> List[ROIConsistencyResult]:
    """Apply ROI consistency scoring to a batch of heatmap/mask pairs."""
    return [
        compute_roi_consistency(h, m, outside_threshold=outside_threshold)
        for h, m in zip(heatmaps, roi_masks)
    ]
