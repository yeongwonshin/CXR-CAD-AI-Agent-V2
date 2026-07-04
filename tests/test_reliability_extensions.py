from __future__ import annotations

import numpy as np

from src.reliability import compute_roi_consistency, detect_hidden_strata, build_readiness_report


def test_roi_consistency_flags_shortcut_activation():
    heatmap = np.zeros((4, 4), dtype=np.float32)
    heatmap[1:3, 1:3] = 1.0
    heatmap[0, 0] = 10.0
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[1:3, 1:3] = 1.0

    result = compute_roi_consistency(heatmap, mask, outside_threshold=0.4)

    assert result.shortcut_flag is True
    assert result.outside_energy_ratio > 0.4
    assert result.roi_pixels == 4


def test_hidden_stratification_returns_flagged_count():
    rng = np.random.default_rng(7)
    x_good = rng.normal(0, 0.2, size=(40, 3))
    x_bad = rng.normal(4, 0.2, size=(40, 3))
    embeddings = np.vstack([x_good, x_bad])
    y = np.array([0, 1] * 40)
    p = np.r_[np.where(y[:40] == 1, 0.9, 0.1), np.where(y[40:] == 1, 0.1, 0.9)]

    result = detect_hidden_strata(embeddings, y, p, n_clusters=2, min_size=10)

    assert result["flagged_count"] >= 1
    assert len(result["strata"]) == 2


def test_readiness_report_blocks_critical_issue():
    report = build_readiness_report(external_drop_pp=4.3, roi_outside_ratio=0.55)

    assert report["overall_status"] == "critical"
    assert len(report["issues"]) == 2
