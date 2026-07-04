"""Cross-dimensional deployment readiness report builder."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional

import numpy as np


@dataclass(frozen=True)
class ReadinessIssue:
    dimension: str
    severity: str
    message: str
    recommended_action: str


def _severity(has_critical: bool, has_warning: bool) -> str:
    if has_critical:
        return "critical"
    if has_warning:
        return "warning"
    return "pass"


def build_readiness_report(
    calibration_ece: Optional[float] = None,
    youden_j: Optional[float] = None,
    domain_gap_pp: Optional[float] = None,
    external_drop_pp: Optional[float] = None,
    shortcut_ratio: Optional[float] = None,
    roi_outside_ratio: Optional[float] = None,
    hidden_strata_flagged: Optional[int] = None,
) -> Dict[str, object]:
    """Create a compact readiness report from the paper's three dimensions.

    Thresholds follow the paper's rubric where possible: ECE 0.05, Youden J 0.6,
    subgroup/external gap 3 percentage points, and shortcut ratio 5%.
    Added checks strengthen the weak points noted in the conclusion: ROI-based
    explanation consistency and hidden stratification.
    """
    issues: List[ReadinessIssue] = []

    if calibration_ece is not None and calibration_ece >= 0.05:
        issues.append(ReadinessIssue("calibration", "warning", f"ECE={calibration_ece:.3f} exceeds 0.05", "Re-run temperature scaling and reset operating thresholds."))
    if youden_j is not None and youden_j < 0.60:
        issues.append(ReadinessIssue("calibration", "warning", f"Youden J={youden_j:.3f} is below 0.60", "Review disease-specific threshold policy."))
    if domain_gap_pp is not None and domain_gap_pp >= 3.0:
        issues.append(ReadinessIssue("domain_robustness", "warning", f"Subgroup AUROC gap={domain_gap_pp:.1f}pp", "Collect or reweight weak subgroup cases."))
    if external_drop_pp is not None and external_drop_pp >= 3.0:
        issues.append(ReadinessIssue("domain_robustness", "critical", f"External AUROC drop={external_drop_pp:.1f}pp", "Run external-site fine-tuning or domain adaptation before deployment."))
    if shortcut_ratio is not None and shortcut_ratio >= 0.05:
        issues.append(ReadinessIssue("localization", "warning", f"Shortcut pattern ratio={shortcut_ratio:.1%}", "Clean shortcut-prone samples and verify with ROI consistency."))
    if roi_outside_ratio is not None and roi_outside_ratio >= 0.40:
        issues.append(ReadinessIssue("localization", "critical", f"Explanation energy outside ROI={roi_outside_ratio:.1%}", "Add lung/lesion ROI mask checks and retrain with ROI-aware constraints."))
    if hidden_strata_flagged is not None and hidden_strata_flagged > 0:
        issues.append(ReadinessIssue("hidden_stratification", "warning", f"{hidden_strata_flagged} underperforming hidden strata detected", "Review cluster exemplars and add targeted validation slices."))

    has_critical = any(i.severity == "critical" for i in issues)
    has_warning = any(i.severity == "warning" for i in issues)
    overall = _severity(has_critical, has_warning)
    return {
        "overall_status": overall,
        "deployment_recommendation": {
            "pass": "Proceed with routine monitoring.",
            "warning": "Do not fully deploy until warning items are reviewed.",
            "critical": "Block deployment until critical reliability issues are resolved.",
        }[overall],
        "issues": [asdict(i) for i in issues],
    }
