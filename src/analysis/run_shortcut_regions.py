"""Generate model-specific Grad-CAM shortcut region summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import src.train.models as model_defs
from src.analysis.gradcam import GradCAM, get_target_layer
from src.preprocess.transforms import get_inference_transforms
from src.train.models import DISEASE_LABELS, build_model


MODEL_KEYS = ["densenet", "efficientnet", "vit"]


def _build_image_index(data_dir: Path) -> dict[str, Path]:
    return {path.name: path for path in data_dir.glob("**/*.png")}


def _load_model(model_key: str, checkpoint_dir: Path, device: torch.device) -> torch.nn.Module:
    if model_key == "vit":
        # The local ViT checkpoint uses torchvision ViT state_dict keys.
        model_defs._TIMM_AVAILABLE = False
    model = build_model(model_key)
    ckpt_path = checkpoint_dir / model_key / f"{model_key}_best.pth"
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _lung_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (int(w * 0.34), int(h * 0.53)), (int(w * 0.23), int(h * 0.36)), 0, 0, 360, 1, -1)
    cv2.ellipse(mask, (int(w * 0.66), int(h * 0.53)), (int(w * 0.23), int(h * 0.36)), 0, 0, 360, 1, -1)
    return mask.astype(bool)


def _classify_region(cam: np.ndarray) -> tuple[str, float, tuple[int, int]]:
    h, w = cam.shape
    mask = _lung_mask((h, w))
    total = float(cam.sum())
    lung_ratio = float((cam * mask.astype(float)).sum() / max(total, 1e-8))
    peak_y, peak_x = np.unravel_index(int(cam.argmax()), cam.shape)
    x = peak_x / max(w - 1, 1)
    y = peak_y / max(h - 1, 1)
    peak_in_lung = bool(mask[peak_y, peak_x])

    if peak_in_lung and lung_ratio >= 0.45:
        return "Lungs (expected)", lung_ratio, (int(peak_y), int(peak_x))
    if y < 0.18 and (x < 0.30 or x > 0.70):
        return "Markers/Text", lung_ratio, (int(peak_y), int(peak_x))
    if y < 0.30:
        return "Bones/Clavicles", lung_ratio, (int(peak_y), int(peak_x))
    if y > 0.78:
        return "Diaphragm/Upper Abdomen", lung_ratio, (int(peak_y), int(peak_x))
    if 0.40 <= x <= 0.62 and 0.30 <= y <= 0.82:
        return "Heart/Mediastinum", lung_ratio, (int(peak_y), int(peak_x))
    if x < 0.10 or x > 0.90 or y < 0.08 or y > 0.92:
        return "Image Border", lung_ratio, (int(peak_y), int(peak_x))
    return "Outside Lung", lung_ratio, (int(peak_y), int(peak_x))


def _case_disease(row: pd.Series, error_type: str) -> str | None:
    value = str(row["예측"] if error_type == "FP" else row["GT"]).strip()
    if value in DISEASE_LABELS:
        return value
    return None


def _is_shortcut_region(region: str) -> bool:
    return "lung" not in str(region).lower()


def _count_error_candidates(model_dir: Path) -> int:
    total = 0
    for filename in ["false_positive.csv", "false_negative.csv"]:
        path = model_dir / filename
        if not path.exists():
            continue
        try:
            total += len(pd.read_csv(path))
        except Exception:
            continue
    return total


def _load_cases(model_dir: Path, max_per_type: int) -> pd.DataFrame:
    frames = []
    for filename, error_type in [("false_positive.csv", "FP"), ("false_negative.csv", "FN")]:
        path = model_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path).head(max_per_type).copy()
        df["Error Type"] = error_type
        df["Disease"] = df.apply(lambda row: _case_disease(row, error_type), axis=1)
        df = df[df["Disease"].notna()]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_input(image_path: Path, transform) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def analyze_model(
    model_key: str,
    data_dir: Path,
    checkpoint_dir: Path,
    image_index: dict[str, Path],
    max_per_type: int,
    device: torch.device,
) -> pd.DataFrame:
    model_dir = checkpoint_dir / model_key
    cases = _load_cases(model_dir, max_per_type)
    if cases.empty:
        raise ValueError(f"No FP/FN cases found for {model_key}")

    model = _load_model(model_key, checkpoint_dir, device)
    gradcam = GradCAM(model, get_target_layer(model))
    transform = get_inference_transforms(224)
    rows = []

    for _, row in cases.iterrows():
        image_name = str(row["Image Index"])
        image_path = image_index.get(image_name)
        if image_path is None:
            continue
        disease = str(row["Disease"])
        class_idx = DISEASE_LABELS.index(disease)
        input_tensor = _load_input(image_path, transform).to(device)
        cam = gradcam.generate(input_tensor, class_idx, image_size=(224, 224))
        region, lung_ratio, peak = _classify_region(cam)
        rows.append(
            {
                "Model": model_key,
                "Image Index": image_name,
                "Error Type": row["Error Type"],
                "Disease": disease,
                "Region": region,
                "Lung Activation Ratio": lung_ratio,
                "Peak Y": peak[0],
                "Peak X": peak[1],
            }
        )

    gradcam.remove_hooks()
    detail_df = pd.DataFrame(rows)
    detail_df.to_csv(model_dir / "shortcut_cases.csv", index=False)

    shortcut_df = (
        detail_df.groupby("Region")
        .size()
        .reset_index(name="Count")
        .rename(columns={"Region": "영역"})
        .sort_values("Count", ascending=False)
    )
    shortcut_df.to_csv(model_dir / "shortcut_regions.csv", index=False)

    sampled_cases = int(len(detail_df))
    shortcut_count = int(detail_df["Region"].map(_is_shortcut_region).sum()) if sampled_cases else 0
    error_candidate_count = _count_error_candidates(model_dir)
    denominator = error_candidate_count if error_candidate_count >= sampled_cases else sampled_cases
    summary_df = pd.DataFrame([
        {
            "Model": model_key,
            "Sampled_Cases": sampled_cases,
            "Shortcut_Count": shortcut_count,
            "Error_Candidate_Count": denominator,
            "Shortcut_Ratio_Sampled": shortcut_count / sampled_cases if sampled_cases else np.nan,
            "Shortcut_Ratio_ErrorCandidates": shortcut_count / denominator if denominator else np.nan,
            "Note": "Ratio for readiness dashboard uses Shortcut_Count / Error_Candidate_Count to avoid treating the small sampled region distribution as the full denominator.",
        }
    ])
    summary_df.to_csv(model_dir / "shortcut_summary.csv", index=False)
    return shortcut_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--models", nargs="+", default=MODEL_KEYS, choices=MODEL_KEYS)
    parser.add_argument("--max_per_type", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Indexing images...")
    image_index = _build_image_index(args.data_dir)
    print(f"Indexed {len(image_index):,} PNG images")

    for model_key in args.models:
        shortcut_df = analyze_model(
            model_key=model_key,
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            image_index=image_index,
            max_per_type=args.max_per_type,
            device=device,
        )
        print(f"{model_key}:")
        print(shortcut_df.to_string(index=False))


if __name__ == "__main__":
    main()
