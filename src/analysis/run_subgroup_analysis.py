"""Generate model-specific subgroup AUROC CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.train.models import DISEASE_LABELS


MODEL_KEYS = ["densenet", "efficientnet", "vit"]
AGE_BINS = [0, 40, 60, 120]
AGE_LABELS = ["Under 40", "40-60", "Over 60"]


def _safe_auc(y_true: pd.Series, y_prob: pd.Series) -> float:
    if len(y_true) == 0 or y_true.nunique(dropna=True) < 2:
        return float("nan")
    return float(roc_auc_score(y_true.astype(int), y_prob.astype(float)))


def _mean_valid(values: list[float]) -> float:
    valid = [value for value in values if np.isfinite(value)]
    return float(np.mean(valid)) if valid else float("nan")


def _gap_percent(lhs: float, rhs: float) -> str:
    if not np.isfinite(lhs) or not np.isfinite(rhs):
        return "NaN"
    return f"{(lhs - rhs) * 100:+.1f}%"


def _disease_auc(df: pd.DataFrame, label: str) -> float:
    return _safe_auc(df[f"{label}_true"], df[f"{label}_prob"])


def build_gender_subgroup(df: pd.DataFrame) -> pd.DataFrame:
    male_df = df[df["Patient Gender"] == "M"]
    female_df = df[df["Patient Gender"] == "F"]
    rows = []
    male_values = []
    female_values = []
    for label in DISEASE_LABELS:
        male_auc = _disease_auc(male_df, label)
        female_auc = _disease_auc(female_df, label)
        male_values.append(male_auc)
        female_values.append(female_auc)
        rows.append(
            {
                "Disease": label,
                "Male AUROC": male_auc,
                "Female AUROC": female_auc,
                "Gap": _gap_percent(male_auc, female_auc),
            }
        )

    male_mean = _mean_valid(male_values)
    female_mean = _mean_valid(female_values)
    rows.append(
        {
            "Disease": "Mean",
            "Male AUROC": male_mean,
            "Female AUROC": female_mean,
            "Gap": _gap_percent(male_mean, female_mean),
        }
    )
    return pd.DataFrame(rows)


def build_age_subgroup(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["Patient Age"] = pd.to_numeric(work["Patient Age"], errors="coerce")
    work = work[(work["Patient Age"] >= 0) & (work["Patient Age"] <= 120)].copy()
    work["Age Group"] = pd.cut(work["Patient Age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)

    rows = []
    for group in AGE_LABELS:
        sub_df = work[work["Age Group"] == group]
        aucs = [_disease_auc(sub_df, label) for label in DISEASE_LABELS]
        rows.append(
            {
                "Age Group": group,
                "N": int(len(sub_df)),
                "Mean AUROC": _mean_valid(aucs),
            }
        )
    return pd.DataFrame(rows)


def build_view_subgroup(df: pd.DataFrame) -> pd.DataFrame:
    view_values = [view for view in ["PA", "AP"] if view in set(df["View Position"].dropna())]
    for view in sorted(set(df["View Position"].dropna())):
        if view not in view_values:
            view_values.append(view)

    rows = []
    pa_auc = float("nan")
    for view in view_values:
        sub_df = df[df["View Position"] == view]
        aucs = [_disease_auc(sub_df, label) for label in DISEASE_LABELS]
        mean_auc = _mean_valid(aucs)
        if view == "PA":
            pa_auc = mean_auc
        rows.append(
            {
                "View": view,
                "N": int(len(sub_df)),
                "Mean AUROC": mean_auc,
                "Gap vs PA": "—" if view == "PA" else _gap_percent(mean_auc, pa_auc),
            }
        )
    return pd.DataFrame(rows)


def analyze_model(checkpoint_dir: Path, model_key: str) -> dict[str, pd.DataFrame]:
    model_dir = checkpoint_dir / model_key
    pred_path = model_dir / "test_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    df = pd.read_csv(pred_path)

    outputs = {
        "gender_subgroup.csv": build_gender_subgroup(df),
        "age_subgroup.csv": build_age_subgroup(df),
        "view_subgroup.csv": build_view_subgroup(df),
    }
    for filename, out_df in outputs.items():
        out_df.to_csv(model_dir / filename, index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--models", nargs="+", default=MODEL_KEYS, choices=MODEL_KEYS)
    args = parser.parse_args()

    for model_key in args.models:
        outputs = analyze_model(args.checkpoint_dir, model_key)
        gender_mean = outputs["gender_subgroup.csv"].query("Disease == 'Mean'").iloc[0]
        age_rows = len(outputs["age_subgroup.csv"])
        view_rows = len(outputs["view_subgroup.csv"])
        print(
            f"{model_key}: gender mean M={gender_mean['Male AUROC']:.4f}, "
            f"F={gender_mean['Female AUROC']:.4f}; age rows={age_rows}; view rows={view_rows}"
        )


if __name__ == "__main__":
    main()
