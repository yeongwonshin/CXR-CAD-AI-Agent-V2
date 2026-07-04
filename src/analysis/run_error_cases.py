"""Generate model-specific false-positive and false-negative case CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.train.models import DISEASE_LABELS


MODEL_KEYS = ["densenet", "efficientnet", "vit"]


def _safe_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[idx])
    if not np.isfinite(threshold):
        return 0.5
    return min(max(threshold, 0.0), 1.0)


def _actual_labels(row: pd.Series) -> str:
    labels = [label for label in DISEASE_LABELS if float(row.get(f"{label}_true", 0.0)) >= 0.5]
    return "|".join(labels) if labels else "Normal/Negative"


def _reason(row: pd.Series, threshold: float, error_type: str) -> str:
    margin = abs(float(row["Probability"]) - threshold)
    view = row.get("View Position", "Unknown")
    age = row.get("Patient Age", "Unknown")
    gender = row.get("Patient Gender", "Unknown")
    if error_type == "FP":
        return f"threshold {threshold:.3f}보다 +{margin:.3f} 높음; View={view}, Age={age}, Sex={gender}"
    return f"threshold {threshold:.3f}보다 -{margin:.3f} 낮음; View={view}, Age={age}, Sex={gender}"


def _format_false_positive(rows: pd.DataFrame, max_cases: int) -> pd.DataFrame:
    output = []
    for idx, (_, row) in enumerate(rows.head(max_cases).iterrows(), start=1):
        output.append(
            {
                "Case": f"FP-{idx}",
                "Image Index": row["Image Index"],
                "예측": row["Disease"],
                "GT": row["Actual Labels"],
                "확률": round(float(row["Probability"]), 4),
                "Threshold": round(float(row["Threshold"]), 4),
                "Grad-CAM": "확률 기반 FP 후보",
                "원인": _reason(row, float(row["Threshold"]), "FP"),
            }
        )
    return pd.DataFrame(output)


def _format_false_negative(rows: pd.DataFrame, max_cases: int) -> pd.DataFrame:
    output = []
    for idx, (_, row) in enumerate(rows.head(max_cases).iterrows(), start=1):
        output.append(
            {
                "Case": f"FN-{idx}",
                "Image Index": row["Image Index"],
                "예측": "Below threshold",
                "GT": row["Disease"],
                "확률": round(float(row["Probability"]), 4),
                "Threshold": round(float(row["Threshold"]), 4),
                "Grad-CAM": "확률 기반 FN 후보",
                "원인": _reason(row, float(row["Threshold"]), "FN"),
            }
        )
    return pd.DataFrame(output)


def analyze_model(checkpoint_dir: Path, model_key: str, max_cases: int) -> dict[str, pd.DataFrame]:
    model_dir = checkpoint_dir / model_key
    pred_path = model_dir / "test_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    df = pd.read_csv(pred_path)

    fp_rows = []
    fn_rows = []
    for label in DISEASE_LABELS:
        true_col = f"{label}_true"
        prob_col = f"{label}_prob"
        y_true = df[true_col].astype(int).to_numpy()
        y_prob = df[prob_col].astype(float).to_numpy()
        threshold = _safe_threshold(y_true, y_prob)

        fp = df[(df[true_col] == 0) & (df[prob_col] >= threshold)].copy()
        fp["Disease"] = label
        fp["Probability"] = fp[prob_col].astype(float)
        fp["Threshold"] = threshold
        fp["Margin"] = fp["Probability"] - threshold
        fp["Actual Labels"] = fp.apply(_actual_labels, axis=1)
        fp_rows.append(fp)

        fn = df[(df[true_col] == 1) & (df[prob_col] < threshold)].copy()
        fn["Disease"] = label
        fn["Probability"] = fn[prob_col].astype(float)
        fn["Threshold"] = threshold
        fn["Margin"] = threshold - fn["Probability"]
        fn["Actual Labels"] = fn.apply(_actual_labels, axis=1)
        fn_rows.append(fn)

    fp_all = pd.concat(fp_rows, ignore_index=True).sort_values("Margin", ascending=False)
    fn_all = pd.concat(fn_rows, ignore_index=True).sort_values("Margin", ascending=False)

    false_positive = _format_false_positive(fp_all, max_cases)
    false_negative = _format_false_negative(fn_all, max_cases)
    false_positive.to_csv(model_dir / "false_positive.csv", index=False)
    false_negative.to_csv(model_dir / "false_negative.csv", index=False)
    return {"false_positive.csv": false_positive, "false_negative.csv": false_negative}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--models", nargs="+", default=MODEL_KEYS, choices=MODEL_KEYS)
    parser.add_argument("--max_cases", type=int, default=12)
    args = parser.parse_args()

    for model_key in args.models:
        outputs = analyze_model(args.checkpoint_dir, model_key, args.max_cases)
        print(
            f"{model_key}: FP={len(outputs['false_positive.csv'])}, "
            f"FN={len(outputs['false_negative.csv'])}"
        )


if __name__ == "__main__":
    main()
