"""Deployment-readiness reliability extensions.

These modules add post-hoc checks without changing the existing training,
API, or dashboard code paths.
"""

from .roi_consistency import ROIConsistencyResult, compute_roi_consistency
from .hidden_stratification import HiddenStratumResult, detect_hidden_strata
from .readiness_report import ReadinessIssue, build_readiness_report

__all__ = [
    "ROIConsistencyResult",
    "compute_roi_consistency",
    "HiddenStratumResult",
    "detect_hidden_strata",
    "ReadinessIssue",
    "build_readiness_report",
]
