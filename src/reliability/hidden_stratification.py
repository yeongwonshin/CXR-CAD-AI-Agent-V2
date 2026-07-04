"""Hidden stratification detection for deployment-readiness analysis.

Hidden stratification means that a model can look acceptable on aggregate but
fail on visually or clinically coherent subgroups. This module clusters feature
embeddings and reports strata whose AUROC or error rate is substantially worse
than the global score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    _SKLEARN_AVAILABLE = False


@dataclass(frozen=True)
class HiddenStratumResult:
    stratum_id: int
    size: int
    prevalence: float
    auroc: float
    error_rate: float
    auroc_drop: float
    error_lift: float
    flagged: bool


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def detect_hidden_strata(
    embeddings: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_clusters: int = 4,
    threshold: float = 0.5,
    min_size: int = 20,
    auroc_drop_warn: float = 0.03,
    error_lift_warn: float = 0.10,
    random_state: int = 42,
) -> Dict[str, object]:
    """Cluster embeddings and find underperforming hidden strata.

    Args:
        embeddings: N x D model features or penultimate-layer embeddings.
        y_true: N binary labels for one target disease.
        y_prob: N predicted probabilities for the same target disease.
        n_clusters: number of candidate hidden strata.
        threshold: probability threshold used for error-rate analysis.
        min_size: ignore tiny clusters below this size.
        auroc_drop_warn: flag if global AUROC - stratum AUROC exceeds this.
        error_lift_warn: flag if stratum error rate - global error rate exceeds this.
    """
    if not _SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for hidden stratification detection")

    x = np.asarray(embeddings, dtype=np.float32)
    y = np.asarray(y_true).astype(int).reshape(-1)
    p = np.asarray(y_prob, dtype=np.float32).reshape(-1)
    if x.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if not (len(x) == len(y) == len(p)):
        raise ValueError("embeddings, y_true, and y_prob must have the same length")
    if len(x) < n_clusters:
        raise ValueError("number of samples must be >= n_clusters")

    x_scaled = StandardScaler().fit_transform(x)
    cluster_ids = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit_predict(x_scaled)

    global_auc = _safe_auc(y, p)
    global_error = float(((p >= threshold).astype(int) != y).mean())
    results: List[HiddenStratumResult] = []

    for cid in sorted(np.unique(cluster_ids)):
        idx = cluster_ids == cid
        size = int(idx.sum())
        if size < min_size:
            continue
        auc = _safe_auc(y[idx], p[idx])
        err = float(((p[idx] >= threshold).astype(int) != y[idx]).mean())
        auc_drop = float(global_auc - auc) if np.isfinite(global_auc) and np.isfinite(auc) else float("nan")
        error_lift = float(err - global_error)
        flagged = (np.isfinite(auc_drop) and auc_drop >= auroc_drop_warn) or (error_lift >= error_lift_warn)
        results.append(
            HiddenStratumResult(
                stratum_id=int(cid),
                size=size,
                prevalence=round(float(y[idx].mean()), 6),
                auroc=round(auc, 6) if np.isfinite(auc) else float("nan"),
                error_rate=round(err, 6),
                auroc_drop=round(auc_drop, 6) if np.isfinite(auc_drop) else float("nan"),
                error_lift=round(error_lift, 6),
                flagged=bool(flagged),
            )
        )

    return {
        "global_auroc": round(global_auc, 6) if np.isfinite(global_auc) else float("nan"),
        "global_error_rate": round(global_error, 6),
        "cluster_labels": cluster_ids.tolist(),
        "strata": results,
        "flagged_count": int(sum(r.flagged for r in results)),
    }
