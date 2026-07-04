"""Lightweight MedRAX-style runtime orchestration for the CXR-CAD platform.

This module does **not** train a new model.  It wraps the platform's existing
prediction, Grad-CAM, DICOM and report-draft outputs into a tool-oriented case
workflow: image-quality check, anatomical ROI scaffold, per-image triage,
multi-image comparison and audit trace.  The goal is to add agentic platform
features without changing the learned CXR classifiers.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from PIL import Image, ImageFilter, ImageStat

CRITICAL_LABELS = {"Pneumothorax", "Pneumonia", "Edema", "Effusion", "Cardiomegaly"}
URGENT_LABELS = {"Pneumothorax"}

ANATOMICAL_ROI_TEMPLATE = {
    "right_upper_lung": {"label_kr": "우상폐야", "bbox_ratio": [0.12, 0.18, 0.43, 0.48]},
    "right_lower_lung": {"label_kr": "우하폐야", "bbox_ratio": [0.12, 0.48, 0.43, 0.82]},
    "left_upper_lung": {"label_kr": "좌상폐야", "bbox_ratio": [0.57, 0.18, 0.88, 0.48]},
    "left_lower_lung": {"label_kr": "좌하폐야", "bbox_ratio": [0.57, 0.48, 0.88, 0.82]},
    "cardiomediastinal": {"label_kr": "심장·종격동", "bbox_ratio": [0.36, 0.38, 0.66, 0.83]},
    "costophrenic_angles": {"label_kr": "늑골횡격막각", "bbox_ratio": [0.08, 0.72, 0.92, 0.94]},
}

DISEASE_TO_ROIS = {
    "Pneumothorax": ["right_upper_lung", "left_upper_lung", "right_lower_lung", "left_lower_lung"],
    "Pneumonia": ["right_lower_lung", "left_lower_lung", "right_upper_lung", "left_upper_lung"],
    "Effusion": ["costophrenic_angles", "right_lower_lung", "left_lower_lung"],
    "Cardiomegaly": ["cardiomediastinal"],
    "Edema": ["cardiomediastinal", "right_lower_lung", "left_lower_lung"],
    "Atelectasis": ["right_lower_lung", "left_lower_lung"],
    "Consolidation": ["right_lower_lung", "left_lower_lung", "right_upper_lung", "left_upper_lung"],
    "Mass": ["right_upper_lung", "left_upper_lung", "right_lower_lung", "left_lower_lung"],
    "Nodule": ["right_upper_lung", "left_upper_lung", "right_lower_lung", "left_lower_lung"],
    "Fibrosis": ["right_upper_lung", "left_upper_lung", "right_lower_lung", "left_lower_lung"],
    "Pleural_Thickening": ["costophrenic_angles", "right_upper_lung", "left_upper_lung"],
    "Emphysema": ["right_upper_lung", "left_upper_lung"],
    "Infiltration": ["right_lower_lung", "left_lower_lung", "right_upper_lung", "left_upper_lung"],
    "Hernia": ["costophrenic_angles", "cardiomediastinal"],
}

DISEASE_KR = {
    "Atelectasis": "무기폐",
    "Cardiomegaly": "심비대",
    "Effusion": "흉수",
    "Infiltration": "폐 침윤",
    "Mass": "종괴",
    "Nodule": "결절",
    "Pneumonia": "폐렴",
    "Pneumothorax": "기흉",
    "Consolidation": "경화",
    "Edema": "폐부종",
    "Emphysema": "폐기종",
    "Fibrosis": "섬유화",
    "Pleural_Thickening": "흉막 비후",
    "Hernia": "탈장",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _entropy(gray: Image.Image) -> float:
    hist = np.array(gray.histogram(), dtype=np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    probs = hist / total
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def analyze_image_quality(image: Image.Image, *, is_dicom_input: bool = False) -> Dict[str, Any]:
    """Return deterministic image-quality indicators for runtime review."""
    rgb = image.convert("RGB")
    gray = rgb.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    stat = ImageStat.Stat(gray)
    width, height = rgb.size
    brightness = float(stat.mean[0]) / 255.0
    contrast = float(stat.stddev[0]) / 255.0
    entropy = _entropy(gray)

    # Variance of a simple high-pass image as a no-opencv sharpness proxy.
    edges = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    sharpness = float(edges.var() / 255.0)
    aspect_ratio = width / max(height, 1)
    dark_fraction = float((arr < 8).mean())
    bright_fraction = float((arr > 247).mean())

    flags: list[str] = []
    if width < 256 or height < 256:
        flags.append("해상도가 낮아 미세 병변 검토에 제한이 있을 수 있음")
    if brightness < 0.18:
        flags.append("전반적으로 어두운 영상")
    elif brightness > 0.82:
        flags.append("전반적으로 밝은 영상")
    if contrast < 0.11:
        flags.append("대비가 낮아 경계 판단이 어려울 수 있음")
    if sharpness < 3.0:
        flags.append("선예도가 낮아 재촬영 또는 원본 확인 권장")
    if dark_fraction > 0.45 or bright_fraction > 0.18:
        flags.append("검은/흰 영역 비율이 높아 crop 또는 windowing 확인 필요")
    if not (0.55 <= aspect_ratio <= 1.45):
        flags.append("흉부 X-ray 표준 비율에서 벗어난 입력일 수 있음")

    quality_score = 100
    quality_score -= 22 if width < 256 or height < 256 else 0
    quality_score -= 18 if contrast < 0.11 else 0
    quality_score -= 12 if brightness < 0.18 or brightness > 0.82 else 0
    quality_score -= 12 if sharpness < 3.0 else 0
    quality_score -= 8 if dark_fraction > 0.45 or bright_fraction > 0.18 else 0
    quality_score -= 6 if not (0.55 <= aspect_ratio <= 1.45) else 0
    quality_score = max(0, min(100, quality_score))

    if quality_score >= 82:
        grade = "양호"
    elif quality_score >= 62:
        grade = "주의"
    else:
        grade = "재확인 필요"

    return {
        "width": width,
        "height": height,
        "aspect_ratio": round(aspect_ratio, 3),
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "entropy": round(entropy, 3),
        "sharpness_proxy": round(sharpness, 3),
        "dark_fraction": round(dark_fraction, 3),
        "bright_fraction": round(bright_fraction, 3),
        "quality_score": quality_score,
        "quality_grade": grade,
        "is_dicom_input": is_dicom_input,
        "flags": flags or ["자동 품질 점검상 큰 제한 신호는 없습니다."],
    }


def build_anatomy_assessment(
    image: Image.Image,
    probs: Mapping[str, float],
    detected: Sequence[str],
    top_disease: str,
) -> Dict[str, Any]:
    """Build a MedRAX-like anatomical review scaffold without new training.

    It intentionally labels the output as a coarse ROI scaffold, not a learned
    segmentation mask.  The boxes help clinicians know which anatomical zones
    to review next while the original Grad-CAM remains the model-attribution
    visualization.
    """
    width, height = image.size
    focus_labels = list(dict.fromkeys([top_disease, *detected]))[:4]
    focus_roi_keys: list[str] = []
    for label in focus_labels:
        for roi in DISEASE_TO_ROIS.get(label, []):
            if roi not in focus_roi_keys:
                focus_roi_keys.append(roi)
    if not focus_roi_keys:
        focus_roi_keys = ["right_lower_lung", "left_lower_lung", "cardiomediastinal"]

    rois = []
    for key in focus_roi_keys[:6]:
        tpl = ANATOMICAL_ROI_TEMPLATE[key]
        x1, y1, x2, y2 = tpl["bbox_ratio"]
        bbox_px = [int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)]
        related = [d for d in focus_labels if key in DISEASE_TO_ROIS.get(d, [])]
        top_related = max(related, key=lambda d: probs.get(d, 0.0)) if related else top_disease
        rois.append(
            {
                "roi_key": key,
                "label_kr": tpl["label_kr"],
                "bbox_ratio": tpl["bbox_ratio"],
                "bbox_px": bbox_px,
                "related_findings": related,
                "priority_score": round(max([_safe_float(probs.get(d, 0.0)) for d in related] or [_safe_float(probs.get(top_disease, 0.0))]), 4),
                "review_hint": f"{tpl['label_kr']}에서 {DISEASE_KR.get(top_related, top_related)} 관련 소견과 원본 음영을 함께 확인하십시오.",
            }
        )

    return {
        "method": "coarse_anatomical_roi_scaffold",
        "disclaimer": "학습된 segmentation mask가 아니라, 예측 질환과 표준 흉부 해부학 위치를 연결한 검토용 ROI 스캐폴드입니다.",
        "focus_rois": rois,
        "recommended_review_order": [r["label_kr"] for r in sorted(rois, key=lambda item: item["priority_score"], reverse=True)],
    }


def build_triage_assessment(
    probs: Mapping[str, float],
    detected: Sequence[str],
    top_disease: str,
    threshold: float,
    quality: Mapping[str, Any],
    is_placeholder: bool,
) -> Dict[str, Any]:
    top_prob = _safe_float(probs.get(top_disease, 0.0))
    urgent_hit = any(_safe_float(probs.get(label, 0.0)) >= threshold for label in URGENT_LABELS)
    critical_hits = [label for label in CRITICAL_LABELS if _safe_float(probs.get(label, 0.0)) >= threshold]
    low_quality = str(quality.get("quality_grade")) == "재확인 필요"

    if is_placeholder:
        level = "DEMO_ONLY"
        label_kr = "시연용 결과"
        reason = "선택 모델 체크포인트가 없어 Placeholder 응답이므로 임상 우선순위로 사용하지 않습니다."
    elif urgent_hit:
        level = "URGENT_REVIEW"
        label_kr = "긴급 검토"
        reason = "기흉 가능성이 임계값 이상으로 표시되어 즉시 원본 영상과 임상 상태를 확인해야 합니다."
    elif top_prob >= 0.75 or len(critical_hits) >= 2:
        level = "HIGH_PRIORITY"
        label_kr = "우선 검토"
        reason = "상위 확률이 높거나 주요 흉부 소견이 복수로 감지되어 판독 우선순위를 높입니다."
    elif detected or low_quality:
        level = "ROUTINE_REVIEW"
        label_kr = "일반 검토"
        reason = "임계값 이상 소견 또는 영상 품질 제한이 있어 의료진 확인이 필요합니다."
    else:
        level = "LOW_SIGNAL"
        label_kr = "낮은 신호"
        reason = "현재 임계값 이상 소견은 없지만 최종 판독은 원본과 임상정보 기반으로 확인해야 합니다."

    return {
        "triage_level": level,
        "triage_label_kr": label_kr,
        "reason": reason,
        "critical_findings": critical_hits,
        "top_probability": round(top_prob, 4),
        "quality_grade": quality.get("quality_grade"),
    }


def build_agent_case_profile(
    *,
    image: Image.Image,
    filename: str,
    probs: Mapping[str, float],
    detected: Sequence[str],
    top_disease: str,
    threshold: float,
    is_placeholder: bool,
    dicom_metadata: Optional[Mapping[str, Any]] = None,
    is_dicom_input: bool = False,
) -> Dict[str, Any]:
    quality = analyze_image_quality(image, is_dicom_input=is_dicom_input)
    anatomy = build_anatomy_assessment(image, probs, detected, top_disease)
    triage = build_triage_assessment(probs, detected, top_disease, threshold, quality, is_placeholder)
    return {
        "filename": filename,
        "quality_check": quality,
        "anatomy_assessment": anatomy,
        "triage_assessment": triage,
        "dicom_metadata": dict(dicom_metadata or {}),
        "agent_summary": _case_summary_text(filename, top_disease, probs, detected, triage, quality, is_placeholder),
    }


def _case_summary_text(
    filename: str,
    top_disease: str,
    probs: Mapping[str, float],
    detected: Sequence[str],
    triage: Mapping[str, Any],
    quality: Mapping[str, Any],
    is_placeholder: bool,
) -> str:
    top_kr = DISEASE_KR.get(top_disease, top_disease)
    top_prob = _safe_float(probs.get(top_disease, 0.0)) * 100
    detected_kr = ", ".join(DISEASE_KR.get(d, d) for d in detected) if detected else "없음"
    prefix = "[시연용] " if is_placeholder else ""
    return (
        f"{prefix}{filename}: Top 소견은 {top_kr}({top_prob:.1f}%)이며, "
        f"임계값 이상 소견은 {detected_kr}입니다. "
        f"Agent 판정은 {triage.get('triage_label_kr')}이고, "
        f"영상 품질은 {quality.get('quality_grade')}입니다."
    )


def _top_items(case: Mapping[str, Any], limit: int = 5) -> list[tuple[str, float]]:
    probs = case.get("probabilities") or {}
    if not isinstance(probs, Mapping):
        return []
    return sorted(((str(k), _safe_float(v)) for k, v in probs.items()), key=lambda item: item[1], reverse=True)[:limit]


def _case_overview_rows(cases: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, case in enumerate(cases):
        profile = case.get("agent_profile") or {}
        triage = profile.get("triage_assessment") or {}
        quality = profile.get("quality_check") or {}
        top_items = _top_items(case, limit=3)
        rows.append(
            {
                "index": idx + 1,
                "filename": case.get("filename"),
                "case_id": case.get("case_id"),
                "top_disease": case.get("top_disease"),
                "top_disease_kr": DISEASE_KR.get(str(case.get("top_disease")), str(case.get("top_disease"))),
                "top_probability": round(_safe_float(case.get("top_probability")), 4),
                "top3_probabilities": [
                    {"label": label, "label_kr": DISEASE_KR.get(label, label), "probability": round(prob, 4)}
                    for label, prob in top_items
                ],
                "detected_diseases": list(case.get("detected_diseases") or [])[:8],
                "triage_label_kr": triage.get("triage_label_kr"),
                "quality_grade": quality.get("quality_grade"),
                "quality_score": quality.get("quality_score"),
                "is_placeholder": bool(case.get("is_placeholder")),
            }
        )
    return rows


def _probability_matrix_rows(cases: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    labels = sorted({label for case in cases for label in ((case.get("probabilities") or {}).keys())})
    rows: List[Dict[str, Any]] = []
    for label in labels:
        row: Dict[str, Any] = {"label": label, "label_kr": DISEASE_KR.get(label, label)}
        for idx, case in enumerate(cases):
            probs = case.get("probabilities") or {}
            row[f"case_{idx + 1}"] = round(_safe_float(probs.get(label, 0.0)), 4)
        rows.append(row)
    return rows


def build_case_comparison(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(cases) < 2:
        return {
            "enabled": False,
            "summary": "비교 분석은 2장 이상 업로드했을 때 활성화됩니다.",
            "probability_deltas": [],
            "case_overview": _case_overview_rows(cases),
            "probability_matrix": _probability_matrix_rows(cases),
        }

    first = cases[0]
    last = cases[-1]
    first_probs = first.get("probabilities") or {}
    last_probs = last.get("probabilities") or {}
    labels = sorted(set(first_probs.keys()) | set(last_probs.keys()))
    deltas = []
    for label in labels:
        before = _safe_float(first_probs.get(label, 0.0))
        after = _safe_float(last_probs.get(label, 0.0))
        deltas.append(
            {
                "label": label,
                "label_kr": DISEASE_KR.get(label, label),
                "first_probability": round(before, 4),
                "last_probability": round(after, 4),
                "delta": round(after - before, 4),
            }
        )
    deltas_sorted = sorted(deltas, key=lambda item: abs(item["delta"]), reverse=True)
    rising = [d for d in sorted(deltas, key=lambda item: item["delta"], reverse=True) if d["delta"] > 0][:3]
    falling = [d for d in sorted(deltas, key=lambda item: item["delta"]) if d["delta"] < 0][:3]
    first_name = str(first.get("filename", "첫 번째 영상"))
    last_name = str(last.get("filename", "마지막 영상"))
    if rising:
        rising_txt = ", ".join(f"{d['label_kr']} {d['delta'] * 100:+.1f}%p" for d in rising)
    else:
        rising_txt = "뚜렷한 증가 없음"
    if falling:
        falling_txt = ", ".join(f"{d['label_kr']} {d['delta'] * 100:+.1f}%p" for d in falling)
    else:
        falling_txt = "뚜렷한 감소 없음"
    summary = (
        f"{first_name} 대비 {last_name}에서 증가한 주요 신호: {rising_txt}. "
        f"감소한 주요 신호: {falling_txt}. 동일 환자 추적 영상이라면 촬영 조건과 시간 간격을 함께 확인하십시오."
    )
    return {
        "enabled": True,
        "summary": summary,
        "probability_deltas": deltas_sorted[:8],
        "rising_findings": rising,
        "falling_findings": falling,
        "case_overview": _case_overview_rows(cases),
        "probability_matrix": _probability_matrix_rows(cases),
    }


def build_agent_batch_summary(
    cases: Sequence[Mapping[str, Any]], *, question: str = "") -> Dict[str, Any]:
    case_count = len(cases)
    placeholder_count = sum(1 for case in cases if case.get("is_placeholder"))
    triage_counts: Dict[str, int] = {}
    for case in cases:
        triage = (case.get("agent_profile") or {}).get("triage_assessment") or {}
        label = str(triage.get("triage_label_kr", "미분류"))
        triage_counts[label] = triage_counts.get(label, 0) + 1

    top_cases = sorted(
        cases,
        key=lambda case: _safe_float(case.get("top_probability", 0.0)),
        reverse=True,
    )[:3]
    top_case_text = ", ".join(
        f"{case.get('filename', '-')}: {DISEASE_KR.get(str(case.get('top_disease')), str(case.get('top_disease')))} "
        f"{_safe_float(case.get('top_probability', 0.0)) * 100:.1f}%"
        for case in top_cases
    ) or "없음"

    comparison = build_case_comparison(cases)
    if case_count == 0:
        narrative = "분석된 영상이 없습니다."
    else:
        narrative = (
            f"Agent가 {case_count}개 영상을 순차 분석했습니다. "
            f"Top 우선 검토 케이스는 {top_case_text}입니다. "
            f"Triage 분포는 {triage_counts}입니다."
        )
        if placeholder_count:
            narrative += f" 단, {placeholder_count}개 케이스는 Placeholder 응답이므로 임상 판정으로 사용하면 안 됩니다."
        if comparison.get("enabled"):
            narrative += " " + str(comparison.get("summary"))
        if question.strip():
            narrative += f" 사용자 질문 '{question.strip()}'에 대해서는 위 예측·품질·비교 결과 범위 안에서 검토하십시오."

    return {
        "case_count": case_count,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question": question.strip(),
        "triage_counts": triage_counts,
        "placeholder_count": placeholder_count,
        "top_cases": [
            {
                "filename": case.get("filename"),
                "case_id": case.get("case_id"),
                "top_disease": case.get("top_disease"),
                "top_probability": case.get("top_probability"),
            }
            for case in top_cases
        ],
        "comparison": comparison,
        "narrative": narrative,
        "safety_note": "본 Agent 결과는 판독 보조 워크플로우이며 최종 진단이 아닙니다. 의료진이 원본 영상, 과거 영상, 임상정보를 함께 확인해야 합니다.",
    }


def build_tool_trace(*, case_count: int, include_dicom: bool = False, include_comparison: bool = False) -> List[Dict[str, Any]]:
    tools = [
        ("InputRouter", "입력 파일 형식 확인 및 DICOM/일반 이미지 경로 분기"),
        ("CXRClassifier", "기존 학습 모델 또는 Placeholder로 14개 흉부 질환 확률 산출"),
        ("ReportDraftTool", "기존 판독문 초안 생성 로직으로 Findings/Impression 작성"),
        ("GradCAMTool", "실제 모델 가중치가 있을 때 Top 소견 기준 활성화 맵 생성"),
        ("QualityCheckTool", "해상도·밝기·대비·선예도 기반 영상 품질 점검"),
        ("AnatomicalROITool", "질환별 표준 해부학 위치를 이용한 검토 ROI 스캐폴드 생성"),
        ("TriageTool", "Top 확률·중요 질환·품질 제한·Placeholder 여부를 통합해 우선순위 산정"),
    ]
    if include_dicom:
        tools.insert(1, ("DICOMTool", "DICOM windowing 및 주요 메타데이터 추출"))
    if include_comparison:
        tools.append(("ComparisonTool", "여러 영상 간 질환 확률 변화량과 우선 검토 변화 계산"))

    return [
        {
            "step": idx + 1,
            "tool": name,
            "description": desc,
            "status": "completed",
            "case_count": case_count,
        }
        for idx, (name, desc) in enumerate(tools)
    ]
