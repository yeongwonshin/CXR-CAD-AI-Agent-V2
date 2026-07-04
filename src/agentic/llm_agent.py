"""LLM-first MedRAX-style answer engine for the CXR-CAD workbench.

The trained CXR models remain unchanged. This module makes follow-up chat feel
like an agent instead of a fixed FAQ layer: a planner chooses which existing
CXR-CAD tool contexts are relevant, a trace is emitted for the UI, and an
OpenAI-compatible LLM synthesizes the final answer from those observations.
When no LLM endpoint is configured, the deterministic fallback remains grounded
in the same tool observations but explicitly marks missing data instead of
pretending a tool produced a result.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _load_project_env() -> None:
    """Load .env so backend LLM agent can read local API configuration."""
    try:
        from dotenv import load_dotenv
    except Exception:  # pragma: no cover - optional dependency guard
        return

    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    seen: set[Path] = set()
    for dotenv_path in candidates:
        dotenv_path = dotenv_path.resolve()
        if dotenv_path in seen:
            continue
        seen.add(dotenv_path)
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path, override=False)


_load_project_env()

TOOL_CONTEXT_NAMES = [
    "InputRouter",
    "DICOMTool",
    "CXRClassifierTool",
    "ReportDraftTool",
    "GradCAMTool",
    "QualityCheckTool",
    "AnatomicalROITool",
    "TriageTool",
    "ComparisonTool",
]

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


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    try:
        return dict(value)
    except Exception:
        return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _truncate_text(value: Any, limit: int = 700) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return default


def _llm_configured() -> bool:
    _load_project_env()
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("CXR_AGENT_LLM_API_KEY"))




def get_agent_runtime_status() -> Dict[str, Any]:
    """Return non-secret runtime settings for the Workbench UI.

    This lets the platform distinguish a real LLM-first agent run from the
    deterministic grounded fallback during demos. API keys are never returned.
    """
    _load_project_env()
    configured = _llm_configured()
    model = os.getenv("CXR_AGENT_LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("CXR_AGENT_LLM_BASE_URL") or "https://api.openai.com/v1"
    return {
        "llm_configured": configured,
        "llm_enabled": _env_bool("CXR_AGENT_LLM_ENABLED", True),
        "llm_first": _env_bool("CXR_AGENT_LLM_FIRST", True),
        "llm_planner_enabled": _env_bool("CXR_AGENT_LLM_PLANNER_ENABLED", True),
        "tool_first_fastpath": _env_bool("CXR_AGENT_TOOL_FIRST_FASTPATH", False),
        "model": model if configured else None,
        "base_url": base_url.rsplit("/", 1)[0] + "/..." if configured else None,
        "timeout_seconds": _env_int("CXR_AGENT_LLM_TIMEOUT", 75),
        "max_tokens": _env_int("CXR_AGENT_LLM_MAX_TOKENS", 900),
        "mode_label": "LLM-first MedRAX-style agent" if configured else "local grounded fallback",
        "demo_recommendation": (
            "READY: planner + tool trace + LLM synthesis are active."
            if configured
            else "Set OPENAI_API_KEY or CXR_AGENT_LLM_API_KEY to avoid deterministic fallback answers."
        ),
    }


def _clean_missing(value: Any, default: str = "-") -> Any:
    if value is None or value == "" or value == [] or value == {}:
        return default
    return value


def _top_probabilities(probabilities: Mapping[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    items = sorted(
        ((str(k), _safe_float(v)) for k, v in probabilities.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {"label": label, "label_kr": DISEASE_KR.get(label, label), "probability": round(prob, 4)}
        for label, prob in items[:limit]
    ]


def _strip_prediction(prediction: Mapping[str, Any]) -> Dict[str, Any]:
    pred = _as_dict(prediction)
    cleaned: Dict[str, Any] = {}
    for key in [
        "Case_ID",
        "Detected_Diseases",
        "Top_Disease",
        "Top_Probability",
        "Inference_Time_ms",
        "Model_Used",
        "Model_Key",
        "Is_Placeholder",
        "Report_Draft",
        "Findings_KR",
        "Impression_KR",
        "Need_Review_Reason",
        "Image_Metadata",
        "Quality_Check",
        "Anatomy_Assessment",
        "Triage_Assessment",
        "Agent_Summary",
    ]:
        if key in pred:
            cleaned[key] = pred.get(key)
    cleaned["has_gradcam"] = bool(pred.get("GradCAM_Base64"))
    return cleaned


def _compact_quality(value: Any) -> Dict[str, Any]:
    q = _as_dict(value)
    return {
        "quality_grade": q.get("quality_grade"),
        "quality_score": q.get("quality_score"),
        "flags": list(q.get("flags") or [])[:3],
    }


def _compact_triage(value: Any) -> Dict[str, Any]:
    t = _as_dict(value)
    return {
        "triage_level": t.get("triage_level"),
        "triage_label_kr": t.get("triage_label_kr"),
        "reason": _truncate_text(t.get("reason"), 260),
        "priority_findings": list(t.get("priority_findings") or [])[:5],
    }


def _compact_anatomy(value: Any) -> Dict[str, Any]:
    a = _as_dict(value)
    rois = []
    for roi in list(a.get("focus_rois") or [])[:4]:
        r = _as_dict(roi)
        rois.append(
            {
                "label_kr": r.get("label_kr") or r.get("label"),
                "priority_score": r.get("priority_score"),
                "related_findings": list(r.get("related_findings") or [])[:4],
                "review_hint": _truncate_text(r.get("review_hint"), 160),
            }
        )
    return {
        "recommended_review_order": list(a.get("recommended_review_order") or [])[:5],
        "focus_rois": rois,
        "disclaimer": _truncate_text(a.get("disclaimer"), 220),
    }


def _compact_dicom(value: Any) -> Dict[str, Any]:
    meta = _as_dict(value)
    keep = [
        "Modality",
        "BodyPartExamined",
        "ViewPosition",
        "PatientPosition",
        "PhotometricInterpretation",
        "PixelSpacing",
        "StudyDate",
        "SeriesDescription",
        "WindowCenter",
        "WindowWidth",
        "Rows",
        "Columns",
    ]
    return {k: meta.get(k) for k in keep if meta.get(k) not in (None, "", [])}


def compact_agent_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove large image payloads and keep only LLM-safe tool outputs.

    The Streamlit Workbench may already send a compact payload. In that case we
    must not compact it a second time, because the nested `prediction` and
    `agent_profile` objects have intentionally been removed. Double-compaction
    was causing quality/triage/report context to disappear from chat answers.
    """
    result = _as_dict(result)
    existing_cases = list(result.get("cases") or [])
    if existing_cases:
        first = _as_dict(existing_cases[0])
        already_compact = "prediction" not in first and any(
            key in first for key in ["quality_check", "triage_assessment", "report_draft", "top_probabilities"]
        )
        if already_compact:
            summary = _as_dict(result.get("agent_summary"))
            return {
                "status": result.get("status"),
                "model_key": result.get("model_key"),
                "threshold": result.get("threshold"),
                "case_count": result.get("case_count") or len(existing_cases),
                "cases": [_as_dict(case) for case in existing_cases],
                "agent_summary": summary,
                "safety_note": result.get("safety_note") or summary.get("safety_note"),
            }
    compact_cases: List[Dict[str, Any]] = []
    for idx, raw_case in enumerate(result.get("cases", []) or []):
        case = _as_dict(raw_case)
        prediction = _strip_prediction(_as_dict(case.get("prediction")))
        probabilities = _as_dict(case.get("probabilities"))
        profile = _as_dict(case.get("agent_profile"))
        image_meta = _as_dict(prediction.get("Image_Metadata"))

        quality = profile.get("quality_check") or prediction.get("Quality_Check") or case.get("quality_check") or {}
        triage = profile.get("triage_assessment") or prediction.get("Triage_Assessment") or case.get("triage_assessment") or {}
        anatomy = profile.get("anatomy_assessment") or prediction.get("Anatomy_Assessment") or case.get("anatomy_assessment") or {}
        dicom_meta = profile.get("dicom_metadata") or image_meta.get("dicom_metadata") or case.get("dicom_metadata") or {}

        top_disease = case.get("top_disease") or prediction.get("Top_Disease")
        top_probs = case.get("top_probabilities") or _top_probabilities(probabilities)
        compact_cases.append(
            {
                "index": idx + 1,
                "filename": case.get("filename"),
                "case_id": case.get("case_id") or prediction.get("Case_ID"),
                "top_disease": top_disease,
                "top_disease_kr": DISEASE_KR.get(str(top_disease), str(top_disease)),
                "top_probability": case.get("top_probability") or prediction.get("Top_Probability"),
                "detected_diseases": list(case.get("detected_diseases") or prediction.get("Detected_Diseases") or [])[:8],
                "top_probabilities": list(top_probs or [])[:5],
                "probabilities": {str(label): round(_safe_float(prob), 4) for label, prob in probabilities.items()},
                "is_placeholder": case.get("is_placeholder") if case.get("is_placeholder") is not None else prediction.get("Is_Placeholder"),
                "quality_check": _compact_quality(quality),
                "triage_assessment": _compact_triage(triage),
                "anatomy_assessment": _compact_anatomy(anatomy),
                "dicom_metadata": _compact_dicom(dicom_meta),
                "report_draft": _truncate_text(prediction.get("Report_Draft") or case.get("report_draft"), 700),
                "findings_kr": _truncate_text(prediction.get("Findings_KR") or case.get("findings_kr"), 320),
                "impression_kr": _truncate_text(prediction.get("Impression_KR") or case.get("impression_kr"), 320),
                "has_gradcam": bool(case.get("has_gradcam") or prediction.get("has_gradcam")),
            }
        )

    summary = _as_dict(result.get("agent_summary"))
    comparison = _as_dict(summary.get("comparison"))
    deltas = comparison.get("probability_deltas") or comparison.get("largest_changes") or comparison.get("top_changes") or []
    compact_summary = {
        "narrative": _truncate_text(summary.get("narrative"), 700),
        "safety_note": _truncate_text(summary.get("safety_note"), 300),
        "placeholder_count": summary.get("placeholder_count"),
        "comparison": {
            "enabled": comparison.get("enabled", False),
            "summary": _truncate_text(comparison.get("summary"), 500),
            "probability_deltas": list(deltas or [])[:6] if isinstance(deltas, list) else [],
        },
    }
    return {
        "status": result.get("status"),
        "model_key": result.get("model_key"),
        "threshold": result.get("threshold"),
        "case_count": result.get("case_count") or len(compact_cases),
        "agent_summary": compact_summary,
        "safety_note": result.get("safety_note"),
        "cases": compact_cases,
        "available_tool_outputs": TOOL_CONTEXT_NAMES,
    }


def _history_to_messages(history: Sequence[Mapping[str, str]], limit: int = 4) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for item in list(history or [])[-limit:]:
        role = str(item.get("role", "agent")).lower()
        if role in {"assistant", "agent", "cxr agent"}:
            mapped = "assistant"
        elif role in {"user", "you"}:
            mapped = "user"
        else:
            continue
        content = _truncate_text(item.get("content", ""), 700)
        if content:
            messages.append({"role": mapped, "content": content})
    return messages


def _system_prompt() -> str:
    return """
You are the CXR-CAD runtime agent, modeled after MedRAX's LLM-plus-tools workflow.
You have already selected tool contexts and observed their outputs. Use only the
supplied CXR-CAD context. Do not invent disease labels, locations, image quality
grades, Grad-CAM locations, or report findings that are not supported.

Answer in Korean unless the user asks otherwise. Be clinically useful and
show agent-like reasoning in a compact way:
1) state what you used,
2) answer the user's exact question,
3) list case-specific evidence,
4) state missing tool outputs honestly,
5) finish with a brief clinical-safety note.

If quality/triage/report/Grad-CAM fields are missing, do not print '-' as the
answer. Say that the tool output is unavailable in the current context and
explain what can and cannot be concluded from the classifier probabilities.
This is clinical decision support, not final diagnosis.
""".strip()


def _case_context_line(case: Mapping[str, Any]) -> str:
    top_probs = []
    for item in list(case.get("top_probabilities") or [])[:5]:
        label = item.get("label") or item.get("disease") or "-"
        prob = _safe_float(item.get("probability"))
        top_probs.append(f"{DISEASE_KR.get(label, label)} {prob:.1%}")
    quality = _as_dict(case.get("quality_check"))
    triage = _as_dict(case.get("triage_assessment"))
    anatomy = _as_dict(case.get("anatomy_assessment"))
    rois = ", ".join(str(r.get("label_kr") or r.get("label")) for r in list(anatomy.get("focus_rois") or [])[:3])
    return (
        f"Case {case.get('index')}: {case.get('filename')} | "
        f"top={case.get('top_disease_kr')} {_safe_float(case.get('top_probability')):.1%} | "
        f"detected={', '.join(map(str, case.get('detected_diseases') or [])) or '-'} | "
        f"top_probs={'; '.join(top_probs) or '-'} | "
        f"triage={triage.get('triage_label_kr') or '-'} ({_truncate_text(triage.get('reason'), 140)}) | "
        f"quality={quality.get('quality_grade') or '-'} {quality.get('quality_score') or '-'}; flags={', '.join(map(str, quality.get('flags') or [])) or '-'} | "
        f"ROI={rois or '-'} | GradCAM={'yes' if case.get('has_gradcam') else 'no'} | "
        f"Findings={_truncate_text(case.get('findings_kr'), 180)} | Impression={_truncate_text(case.get('impression_kr'), 160)}"
    )


def _build_context_text(compact_result: Mapping[str, Any]) -> str:
    summary = _as_dict(compact_result.get("agent_summary"))
    comparison = _as_dict(summary.get("comparison"))
    lines = [
        f"model={compact_result.get('model_key')} threshold={compact_result.get('threshold')} cases={compact_result.get('case_count')}",
        f"summary={_truncate_text(summary.get('narrative'), 650)}",
    ]
    if comparison.get("enabled"):
        lines.append(f"comparison={_truncate_text(comparison.get('summary'), 420)}")
        for item in list(comparison.get("probability_deltas") or [])[:5]:
            label = item.get("label_kr") or DISEASE_KR.get(str(item.get("label")), str(item.get("label")))
            delta = _safe_float(item.get("delta"))
            lines.append(f"- change {label}: {delta:+.1%}p")
    for case in list(compact_result.get("cases") or [])[:8]:
        lines.append(_case_context_line(_as_dict(case)))
    safety = compact_result.get("safety_note") or summary.get("safety_note")
    if safety:
        lines.append(f"safety={_truncate_text(safety, 220)}")
    return "\n".join(lines)


def _build_llm_messages(question: str, compact_result: Mapping[str, Any], history: Sequence[Mapping[str, str]]) -> List[Dict[str, str]]:
    context_text = _build_context_text(compact_result)
    messages: List[Dict[str, str]] = [{"role": "system", "content": _system_prompt()}]
    messages.extend(_history_to_messages(history))
    messages.append(
        {
            "role": "user",
            "content": (
                "[CXR-CAD compact tool context]\n"
                f"{context_text}\n\n"
                "[User question]\n"
                f"{question.strip()}"
            ),
        }
    )
    return messages


def _openai_compatible_chat(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    _load_project_env()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CXR_AGENT_LLM_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 또는 CXR_AGENT_LLM_API_KEY가 설정되어 있지 않습니다.")

    model = os.getenv("CXR_AGENT_LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("CXR_AGENT_LLM_BASE_URL") or "https://api.openai.com/v1"
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": _safe_float(os.getenv("CXR_AGENT_LLM_TEMPERATURE"), 0.1),
        "max_tokens": _env_int("CXR_AGENT_LLM_MAX_TOKENS", 900),
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    timeout = _env_int("CXR_AGENT_LLM_TIMEOUT", 75)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    answer = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    if not answer:
        raise RuntimeError("LLM 응답이 비어 있습니다.")
    return {"answer": answer, "model": model, "raw_usage": data.get("usage")}


def _case_line(case: Mapping[str, Any]) -> str:
    idx = case.get("index", "-")
    filename = case.get("filename") or f"case-{idx}"
    top = case.get("top_disease") or "-"
    top_kr = case.get("top_disease_kr") or DISEASE_KR.get(str(top), str(top))
    prob = _safe_float(case.get("top_probability"))
    triage = _as_dict(case.get("triage_assessment"))
    quality = _as_dict(case.get("quality_check"))
    triage_label = _clean_missing(triage.get("triage_label_kr"), "미생성")
    quality_grade = _clean_missing(quality.get("quality_grade"), "미생성")
    return f"- {idx}. {filename}: {top_kr}({top}) {prob:.1%}, triage={triage_label}, 품질={quality_grade}"



def _case_probability(case: Mapping[str, Any], label: str) -> float:
    probabilities = _as_dict(case.get("probabilities"))
    if label in probabilities:
        return _safe_float(probabilities.get(label))
    for item in list(case.get("top_probabilities") or []):
        if item.get("label") == label:
            return _safe_float(item.get("probability"))
    return 0.0


def _comparison_focus_labels(cases: Sequence[Mapping[str, Any]], limit: int = 6) -> List[str]:
    scores: Dict[str, float] = {}
    for case in cases or []:
        probabilities = _as_dict(case.get("probabilities"))
        if probabilities:
            for label, prob in probabilities.items():
                scores[str(label)] = max(scores.get(str(label), 0.0), _safe_float(prob))
        else:
            for item in list(case.get("top_probabilities") or []):
                label = str(item.get("label") or "")
                if label:
                    scores[label] = max(scores.get(label, 0.0), _safe_float(item.get("probability")))
    return [label for label, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]]


def _answer_all_case_comparison(cases: Sequence[Mapping[str, Any]], comparison: Mapping[str, Any] | None = None) -> str:
    case_list = list(cases or [])
    if not case_list:
        return "아직 분석된 영상이 없어 여러 영상 비교표를 만들 수 없습니다."

    lines: List[str] = [
        "여러 장이므로 첫 번째와 마지막 영상만 비교하지 않고, 업로드된 모든 영상의 분석 결과를 동시에 정리합니다.",
        "\n영상별 요약:",
    ]
    for case in case_list:
        lines.append(_case_line(case))
        top_probs = _top_probability_lines(case, limit=3)
        if top_probs:
            lines.append("  · 상위 확률: " + "; ".join(top_probs))

    focus_labels = _comparison_focus_labels(case_list, limit=6)
    if focus_labels:
        lines.append("\n주요 질환별 전체 영상 확률 흐름:")
        for label in focus_labels:
            label_kr = DISEASE_KR.get(label, label)
            flow = " → ".join(
                f"{case.get('index') or idx + 1}번 {_case_probability(case, label):.1%}"
                for idx, case in enumerate(case_list)
            )
            lines.append(f"- {label_kr}: {flow}")

    comp = _as_dict(comparison)
    if comp.get("enabled"):
        lines.append("\n첫 영상 ↔ 마지막 영상 변화량은 보조 지표입니다:")
        deltas = comp.get("probability_deltas") or comp.get("largest_changes") or comp.get("top_changes") or []
        for item in list(deltas)[:5] if isinstance(deltas, list) else []:
            disease = item.get("label_kr") or DISEASE_KR.get(str(item.get("label")), str(item.get("label")))
            delta = _safe_float(item.get("delta") or item.get("change"))
            direction = "증가" if delta > 0 else "감소" if delta < 0 else "변화 없음"
            lines.append(f"- {disease}: {direction} {abs(delta):.1%}p")

    lines.append("\n동일 환자의 시간축 영상이면 촬영 자세·노출·간격 차이를 함께 확인하고, 서로 다른 케이스 묶음이면 순위 비교 용도로만 해석하는 것이 안전합니다.")
    return "\n".join(lines)


_ORDINAL_WORDS = {
    1: ["첫 번째", "첫번째", "첫 번", "첫번", "첫째", "첫 영상", "첫 사진"],
    2: ["두 번째", "두번째", "두 번", "두번", "둘째", "두 영상", "두 사진"],
    3: ["세 번째", "세번째", "세 번", "세번", "셋째", "세 영상", "세 사진"],
    4: ["네 번째", "네번째", "네 번", "네번", "넷째"],
    5: ["다섯 번째", "다섯번째", "다섯째"],
    6: ["여섯 번째", "여섯번째", "여섯째"],
    7: ["일곱 번째", "일곱번째", "일곱째"],
    8: ["여덟 번째", "여덟번째", "여덟째"],
    9: ["아홉 번째", "아홉번째", "아홉째"],
    10: ["열 번째", "열번째", "열째"],
}

_DETAIL_REVIEW_TERMS = [
    "주의", "우선", "먼저", "확인", "부분", "어디", "위치", "부위", "봐야", "보아야",
    "해석", "분석", "읽어", "의료인", "의료진", "중요", "위험", "triage", "priority",
    "interpret", "focus", "review", "where", "attention", "concern",
]

_AVAILABILITY_TERMS = [
    "제공", "있어", "있나", "있나요", "없", "생성", "나와", "보여", "가능", "여부",
    "available", "exist", "shown", "generated",
]


def _requested_case_indices(question: str, max_cases: int) -> List[int]:
    """Find 1-based case numbers mentioned in Korean/English user questions."""
    q = (question or "").lower()
    found: List[int] = []

    for match in re.finditer(r"(?:case|image|img|영상|사진|케이스)\s*#?\s*(10|[1-9])", q):
        idx = int(match.group(1))
        if 1 <= idx <= max_cases and idx not in found:
            found.append(idx)
    for match in re.finditer(r"제\s*(10|[1-9])\s*(?:번|번째|째)?", q):
        idx = int(match.group(1))
        if 1 <= idx <= max_cases and idx not in found:
            found.append(idx)
    for match in re.finditer(r"(?<!\d)(10|[1-9])\s*(?:번|번째|째)(?!\d)", q):
        idx = int(match.group(1))
        if 1 <= idx <= max_cases and idx not in found:
            found.append(idx)

    english = {
        1: ["first", "1st"],
        2: ["second", "2nd"],
        3: ["third", "3rd"],
        4: ["fourth", "4th"],
        5: ["fifth", "5th"],
    }
    for idx, words in english.items():
        if idx <= max_cases and any(re.search(rf"\b{re.escape(word)}\b", q) for word in words):
            if idx not in found:
                found.append(idx)

    for idx, words in _ORDINAL_WORDS.items():
        if idx <= max_cases and any(word in q for word in words):
            if idx not in found:
                found.append(idx)

    return found


def _select_cases(question: str, cases: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    case_list = list(cases or [])
    indices = _requested_case_indices(question, len(case_list))
    if not indices:
        return case_list
    selected: List[Mapping[str, Any]] = []
    for idx in indices:
        if 1 <= idx <= len(case_list):
            selected.append(case_list[idx - 1])
    return selected or case_list


def _is_gradcam_question(question_l: str) -> bool:
    return any(k in question_l for k in ["grad", "cam", "grad-cam", "히트맵", "heatmap", "근거"])


def _is_detail_review_question(question_l: str) -> bool:
    return any(k in question_l for k in _DETAIL_REVIEW_TERMS)


def _is_gradcam_availability_question(question_l: str) -> bool:
    # "Grad-CAM 있어?" should be a quick availability answer, but
    # "세 번째 Grad-CAM에서 주의할 부분" must be interpreted as a review-priority question.
    return _is_gradcam_question(question_l) and any(k in question_l for k in _AVAILABILITY_TERMS) and not _is_detail_review_question(question_l)


def _top_probability_lines(case: Mapping[str, Any], limit: int = 4) -> List[str]:
    lines = []
    for item in list(case.get("top_probabilities") or [])[:limit]:
        label = item.get("label") or item.get("disease") or "-"
        label_kr = item.get("label_kr") or DISEASE_KR.get(str(label), str(label))
        lines.append(f"{label_kr} {_safe_float(item.get('probability')):.1%}")
    return lines


def _case_quality_comment(case: Mapping[str, Any]) -> List[str]:
    quality = _as_dict(case.get("quality_check"))
    grade = quality.get("quality_grade")
    score = quality.get("quality_score")
    flags = list(quality.get("flags") or [])
    if grade or score or flags:
        pieces = [f"등급 {grade}" if grade else "등급 미기록", f"점수 {score}/100" if score is not None else "점수 미기록"]
        if flags:
            pieces.append("; ".join(map(str, flags[:3])))
        return ["- 품질 tool 관찰: " + " · ".join(pieces)]
    return [
        "- 품질 tool 관찰: 현재 chat context에 품질 등급/점수가 없습니다.",
        "- 재촬영 판단: 이 정보만으로는 재촬영 필요성을 단정할 수 없습니다. 원본 영상에서 회전, 흡기 부족, 저노출/과노출, motion blur, scapula overlap, 전체 폐야 포함 여부를 확인해야 합니다.",
    ]


def _synth_report_from_case(case: Mapping[str, Any]) -> Dict[str, str]:
    findings = str(case.get("findings_kr") or "").strip()
    impression = str(case.get("impression_kr") or "").strip()
    if findings or impression:
        return {"findings": findings or "제공된 Findings 초안 없음", "impression": impression or "제공된 Impression 초안 없음"}
    top = case.get("top_disease") or "-"
    top_kr = case.get("top_disease_kr") or DISEASE_KR.get(str(top), str(top))
    prob = _safe_float(case.get("top_probability"))
    top_probs = "; ".join(_top_probability_lines(case, limit=4)) or "상위 확률 정보 없음"
    return {
        "findings": f"AI 분류 결과 {top_kr}({top}) 가능성이 가장 높게 산출되었습니다({prob:.1%}). 함께 확인할 상위 예측은 {top_probs}입니다.",
        "impression": f"{top_kr} 가능성을 우선 검토하십시오. 이는 모델 확률 기반 초안이며 원본 영상과 임상 정보로 확정해야 합니다.",
    }


def _format_focus_rois(case: Mapping[str, Any], limit: int = 3) -> List[str]:
    anatomy = _as_dict(case.get("anatomy_assessment"))
    rois = []
    for roi in list(anatomy.get("focus_rois") or [])[:limit]:
        r = _as_dict(roi)
        label = r.get("label_kr") or r.get("label") or "ROI"
        score = _safe_float(r.get("priority_score"))
        hint = _truncate_text(r.get("review_hint"), 150)
        if hint:
            rois.append(f"{label}({score:.1%}) — {hint}")
        else:
            rois.append(f"{label}({score:.1%})")
    if not rois:
        order = anatomy.get("recommended_review_order") or []
        rois.extend(map(str, order[:limit]))
    return rois


def _answer_gradcam_availability(cases: Sequence[Mapping[str, Any]]) -> str:
    lines = ["Grad-CAM 제공 여부입니다."]
    for case in cases:
        lines.append(f"- {case.get('index')}. {case.get('filename')}: {'제공됨' if case.get('has_gradcam') else '없음 또는 Placeholder'}")
    return "\n".join(lines)


def _answer_case_review_priority(question: str, cases: Sequence[Mapping[str, Any]], *, gradcam_context: bool = False) -> str:
    selected = _select_cases(question, cases)
    if not selected:
        return "분석된 케이스가 없어 우선 확인 부위를 특정할 수 없습니다."

    lines: List[str] = []
    if gradcam_context:
        lines.append("질문을 Grad-CAM의 단순 제공 여부가 아니라, 해당 영상에서 의료진이 우선 확인할 부위/소견 질문으로 해석했습니다.")
    else:
        lines.append("의료진 우선 확인 포인트입니다.")

    for case in selected:
        idx = case.get("index")
        filename = case.get("filename") or f"case-{idx}"
        top = case.get("top_disease") or "-"
        top_kr = case.get("top_disease_kr") or DISEASE_KR.get(str(top), str(top))
        prob = _safe_float(case.get("top_probability"))
        triage = _as_dict(case.get("triage_assessment"))
        quality = _as_dict(case.get("quality_check"))
        rois = _format_focus_rois(case)
        top_probs = _top_probability_lines(case)

        lines.append(f"\n{idx}. {filename}")
        if gradcam_context:
            if case.get("has_gradcam"):
                lines.append("- Grad-CAM: 제공됨. 화면의 고활성 영역이 아래 우선 소견/ROI와 실제로 겹치는지 원본 영상에서 확인하십시오.")
            else:
                lines.append("- Grad-CAM: 없음 또는 Placeholder입니다. 따라서 실제 히트맵 위치 판정은 불가하며, 아래 내용은 분류 결과와 ROI 스캐폴드 기반의 우선 검토 가이드입니다.")
        lines.append(f"- 우선 소견: {top_kr}({top}) {prob:.1%}")
        if top_probs:
            lines.append("- 함께 볼 상위 확률: " + "; ".join(top_probs[:4]))
        if triage:
            reason = _truncate_text(triage.get("reason"), 180)
            lines.append(f"- 검토 우선도: {triage.get('triage_label_kr') or '-'} — {reason or '-'}")
        if rois:
            lines.append("- 먼저 확인할 해부학 부위:")
            for roi_line in rois[:3]:
                lines.append(f"  · {roi_line}")
        flags = quality.get("flags") or []
        if flags:
            lines.append("- 품질/아티팩트 확인: " + "; ".join(map(str, flags[:3])))
        impression = case.get("impression_kr") or case.get("findings_kr")
        if impression:
            lines.append(f"- 판독 초안 참고: {_truncate_text(impression, 220)}")

    lines.append("\n※ Grad-CAM은 진단 확정 근거가 아니라 모델 주목 영역입니다. 의료진은 원본 영상, 임상정보, 이전 영상과 함께 확인해야 합니다.")
    return "\n".join(lines)


def fallback_agent_reply(question: str, compact_result: Mapping[str, Any], *, include_notice: bool = True) -> str:
    """Deterministic backup or fast tool-context answer."""
    question_l = (question or "").lower()
    cases = compact_result.get("cases", []) or []
    summary = _as_dict(compact_result.get("agent_summary"))
    lines: List[str] = []

    mentioned = []
    for label, kr in DISEASE_KR.items():
        if label.lower() in question_l or str(kr).lower() in question_l:
            mentioned.append(label)
    if mentioned:
        lines.append("질문에서 언급한 소견을 영상별로 정리했습니다.")
        for label in mentioned[:4]:
            rows = []
            for case in cases:
                found = 0.0
                for item in case.get("top_probabilities") or []:
                    if item.get("label") == label:
                        found = _safe_float(item.get("probability"))
                        break
                rows.append((found, case))
            lines.append(f"\n{DISEASE_KR.get(label, label)} 기준:")
            for prob, case in sorted(rows, key=lambda x: x[0], reverse=True)[:6]:
                lines.append(f"- {case.get('index')}. {case.get('filename')}: {prob:.1%}")
    elif any(k in question_l for k in ["비교", "변화", "악화", "호전", "compare", "change", "worse", "better"]):
        comparison = _as_dict(summary.get("comparison"))
        lines.append(_answer_all_case_comparison(cases, comparison))
    elif any(k in question_l for k in ["품질", "화질", "흐림", "quality", "재촬영"]):
        lines.append("영상 품질과 재촬영 필요성 검토입니다.")
        lines.append("현재 답변은 CXR-CAD tool context에 남아 있는 품질 관찰값만 사용합니다. 품질 tool 출력이 비어 있으면 재촬영 여부는 단정하지 않습니다.")
        for case in _select_cases(question, cases):
            lines.append(f"\n{case.get('index')}. {case.get('filename')}")
            lines.append(f"- 분류 참고: {case.get('top_disease_kr') or DISEASE_KR.get(str(case.get('top_disease')), str(case.get('top_disease')))} {_safe_float(case.get('top_probability')):.1%}")
            lines.extend(_case_quality_comment(case))
    elif _is_gradcam_question(question_l):
        if _is_gradcam_availability_question(question_l):
            lines.append(_answer_gradcam_availability(_select_cases(question, cases)))
        else:
            lines.append(_answer_case_review_priority(question, cases, gradcam_context=True))
    elif any(k in question_l for k in ["우선", "먼저", "응급", "위험", "주의", "의료인", "의료진", "triage", "priority"]):
        lines.append(_answer_case_review_priority(question, cases, gradcam_context=False))
    elif any(k in question_l for k in ["판독", "초안", "소견", "report", "draft"]):
        lines.append("이미지별 판독문 초안입니다. 제공된 report tool 출력이 비어 있으면 classifier 결과를 근거로 보수적으로 합성했습니다.")
        for case in _select_cases(question, cases):
            report = _synth_report_from_case(case)
            lines.append(f"\n{case.get('index')}. {case.get('filename')}\n- Findings: {report['findings']}\n- Impression: {report['impression']}")
    elif any(k in question_l for k in ["roi", "해부학", "위치", "어디", "location"]):
        lines.append("해부학 ROI 스캐폴드 기준 검토 순서입니다. 이는 학습된 segmentation mask가 아니라 검토 보조 위치입니다.")
        for case in _select_cases(question, cases):
            anatomy = _as_dict(case.get("anatomy_assessment"))
            order = anatomy.get("recommended_review_order") or []
            lines.append(f"- {case.get('index')}. {case.get('filename')}: {', '.join(map(str, order[:5])) if order else '-'}")
    else:
        lines.append(str(summary.get("narrative") or "분석 결과 요약입니다."))
        lines.append("\n우선 검토 후보:")
        for case in sorted(cases, key=lambda c: _safe_float(c.get("top_probability")), reverse=True)[:5]:
            lines.append(_case_line(case))

    if include_notice:
        lines.append("\n※ LLM 엔드포인트가 설정되지 않아 로컬 fallback으로 답변했습니다. OPENAI_API_KEY와 CXR_AGENT_LLM_MODEL을 설정하면 MedRAX식 LLM agent 답변으로 전환됩니다.")
    return "\n".join(lines)


def _fastpath_supported(question: str) -> bool:
    q = (question or "").lower()
    keywords = [
        "비교", "변화", "악화", "호전", "compare", "change", "worse", "better",
        "품질", "화질", "흐림", "quality", "재촬영",
        "grad", "cam", "히트맵", "근거",
        "판독", "초안", "소견", "report", "draft",
        "roi", "해부학", "위치", "어디", "location",
        "우선", "응급", "위험", "주의", "확인", "의료인", "의료진", "triage", "먼저", "priority",
    ]
    if any(k in q for k in keywords):
        return True
    return any(label.lower() in q or kr.lower() in q for label, kr in DISEASE_KR.items())



def _rule_plan_chat_context_tools(question: str, compact_result: Mapping[str, Any]) -> Dict[str, Any]:
    """Deterministic backup planner for selected CXR tool contexts."""
    q = (question or "").lower()
    selected: List[str] = []
    reasoning_steps: List[str] = []

    def add(name: str, reason: str) -> None:
        if name not in selected:
            selected.append(name)
            reasoning_steps.append(f"{name}: {reason}")

    add("CXRClassifierTool", "모든 답변에는 영상별 질환 확률과 Top 소견이 기본 근거로 필요합니다.")
    if any(k in q for k in ["비교", "변화", "악화", "호전", "compare", "change", "worse", "better"]):
        add("ComparisonTool", "사용자가 여러 영상 간 변화/비교를 요청했습니다.")
    if any(k in q for k in ["품질", "화질", "흐림", "quality", "재촬영", "artifact", "아티팩트"]):
        add("QualityCheckTool", "사용자가 화질 또는 재촬영 필요성을 물었습니다.")
    if _is_gradcam_question(q):
        add("GradCAMTool", "사용자가 모델 근거/Grad-CAM/히트맵을 요청했습니다.")
        if not _is_gradcam_availability_question(q):
            add("AnatomicalROITool", "Grad-CAM 근거 설명에는 해부학적 검토 위치가 함께 필요합니다.")
    if any(k in q for k in ["roi", "해부학", "위치", "어디", "location", "where", "부위"]):
        add("AnatomicalROITool", "사용자가 의심 위치 또는 해부학적 부위를 물었습니다.")
    if any(k in q for k in ["우선", "먼저", "응급", "긴급", "위험", "triage", "priority"]):
        add("TriageTool", "사용자가 우선순위/위험도를 물었습니다.")
    if any(k in q for k in ["판독", "초안", "소견", "report", "draft", "findings", "impression", "요약"]):
        add("ReportDraftTool", "사용자가 판독문 또는 소견 요약을 요청했습니다.")
    if any(label.lower() in q or kr.lower() in q for label, kr in DISEASE_KR.items()):
        add("CXRClassifierTool", "특정 질환 확률 질의이므로 classifier context를 사용합니다.")

    case_count = len(compact_result.get("cases", []) or [])
    if case_count > 1 and "ComparisonTool" not in selected and any(k in q for k in ["영상", "케이스", "묶음", "전체", "모두"]):
        add("ComparisonTool", "여러 영상 묶음 전체를 묻고 있어 batch comparison context를 보조 근거로 사용합니다.")
    if len(selected) == 1:
        add("TriageTool", "열린 질문에 대해 우선 검토 순위를 함께 제시하기 위해 사용합니다.")
        add("ReportDraftTool", "열린 질문에 대해 판독문 형태의 요약을 보조 근거로 사용합니다.")

    return {
        "planner_mode": "rule_planner",
        "planned_tools": selected,
        "reasoning_steps": reasoning_steps,
        "answer_style": "case_evidence_markdown",
        "missing_context_policy": "state_unavailable_outputs_explicitly",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_count": case_count,
    }


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _build_planner_prompt(question: str, compact_result: Mapping[str, Any]) -> List[Dict[str, str]]:
    deterministic = _rule_plan_chat_context_tools(question, compact_result)
    available = ", ".join(TOOL_CONTEXT_NAMES)
    context = _build_context_text(compact_result)
    return [
        {
            "role": "system",
            "content": (
                "You are a MedRAX-style tool planner for a CXR-CAD workbench. "
                "Return JSON only. Choose the minimum useful tool contexts for the user's question. "
                "Allowed tools: " + available + ". "
                "Do not invent tools. Prefer explicit missing-context handling over fabricated findings."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question.strip()}\n\n"
                f"Rule planner seed: {json.dumps(deterministic, ensure_ascii=False)}\n\n"
                f"Compact context preview:\n{context[:4500]}\n\n"
                "Return JSON with keys: planned_tools (array), reasoning_steps (array of short Korean strings), "
                "answer_style (string), missing_context_policy (string)."
            ),
        },
    ]


def _normalize_plan(plan: Mapping[str, Any], fallback: Mapping[str, Any], mode: str) -> Dict[str, Any]:
    fallback_tools = list(fallback.get("planned_tools") or [])
    raw_tools = plan.get("planned_tools") if isinstance(plan, Mapping) else []
    selected: List[str] = []
    for item in list(raw_tools or []):
        name = str(item).strip()
        if name in TOOL_CONTEXT_NAMES and name not in selected:
            selected.append(name)
    if "CXRClassifierTool" not in selected:
        selected.insert(0, "CXRClassifierTool")
    for name in fallback_tools:
        # Keep deterministic safety tools when the LLM planner returns too little.
        if name in {"QualityCheckTool", "GradCAMTool", "AnatomicalROITool", "ComparisonTool", "ReportDraftTool", "TriageTool"} and name not in selected:
            selected.append(name)
    if not selected:
        selected = fallback_tools or ["CXRClassifierTool", "TriageTool", "ReportDraftTool"]

    reasoning = plan.get("reasoning_steps") if isinstance(plan, Mapping) else None
    if not isinstance(reasoning, list) or not reasoning:
        reasoning = list(fallback.get("reasoning_steps") or [])
    return {
        "planner_mode": mode,
        "planned_tools": selected,
        "reasoning_steps": [str(x) for x in reasoning[:8]],
        "answer_style": str(plan.get("answer_style") or fallback.get("answer_style") or "case_evidence_markdown"),
        "missing_context_policy": str(plan.get("missing_context_policy") or fallback.get("missing_context_policy") or "state_unavailable_outputs_explicitly"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_count": fallback.get("case_count"),
    }


def _plan_chat_context_tools(question: str, compact_result: Mapping[str, Any]) -> Dict[str, Any]:
    """Plan which CXR tool contexts should answer a follow-up.

    LLM planning is enabled by default when an OpenAI-compatible endpoint is
    configured. The deterministic planner remains as the safe fallback.
    """
    fallback = _rule_plan_chat_context_tools(question, compact_result)
    if not (_env_bool("CXR_AGENT_LLM_PLANNER_ENABLED", True) and _env_bool("CXR_AGENT_LLM_ENABLED", True) and _llm_configured()):
        return dict(fallback)
    try:
        llm_plan_text = _openai_compatible_chat(_build_planner_prompt(question, compact_result))["answer"]
        parsed = _extract_json_object(llm_plan_text)
        normalized = _normalize_plan(parsed, fallback, "llm_planner_openai_compatible")
        normalized["raw_llm_plan"] = _truncate_text(llm_plan_text, 1200)
        return normalized
    except Exception as exc:
        recovered = dict(fallback)
        recovered["planner_mode"] = "rule_planner_after_llm_planner_error"
        recovered["planner_error"] = str(exc)
        return recovered


def _chat_tool_trace(planned_tools: Sequence[str], compact_result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    cases = list(compact_result.get("cases", []) or [])
    summary = _as_dict(compact_result.get("agent_summary"))
    trace: List[Dict[str, Any]] = []
    for iteration, tool_name in enumerate(planned_tools, start=1):
        output: Dict[str, Any]
        if tool_name == "CXRClassifierTool":
            output = {
                "case_count": len(cases),
                "top_cases": [
                    {
                        "index": c.get("index"),
                        "filename": c.get("filename"),
                        "top_disease": c.get("top_disease"),
                        "top_probability": c.get("top_probability"),
                    }
                    for c in cases[:6]
                ],
            }
        elif tool_name == "ComparisonTool":
            output = _as_dict(summary.get("comparison"))
        elif tool_name == "QualityCheckTool":
            output = {"quality": [{"index": c.get("index"), "quality_check": c.get("quality_check")} for c in cases[:6]]}
        elif tool_name == "GradCAMTool":
            output = {"gradcam_available": [{"index": c.get("index"), "has_gradcam": c.get("has_gradcam")} for c in cases[:6]]}
        elif tool_name == "AnatomicalROITool":
            output = {"rois": [{"index": c.get("index"), "anatomy_assessment": c.get("anatomy_assessment")} for c in cases[:6]]}
        elif tool_name == "TriageTool":
            output = {"triage": [{"index": c.get("index"), "triage_assessment": c.get("triage_assessment")} for c in cases[:6]]}
        elif tool_name == "ReportDraftTool":
            output = {"reports": [{"index": c.get("index"), "report_draft": _truncate_text(c.get("report_draft"), 220)} for c in cases[:6]]}
        else:
            output = {"status": "selected"}
        trace.append(
            {
                "iteration": iteration,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "name": tool_name,
                "scope": "chat_context",
                "status": "completed",
                "rationale": "Selected by follow-up question intent and available /agent/analyze outputs.",
                "output_summary": output,
            }
        )
    return trace

def _build_answer_messages(
    question: str,
    compact_result: Mapping[str, Any],
    history: Sequence[Mapping[str, str]],
    plan: Mapping[str, Any],
    trace: Sequence[Mapping[str, Any]],
) -> List[Dict[str, str]]:
    context_text = _build_context_text(compact_result)
    plan_text = json.dumps(
        {
            "planner_mode": plan.get("planner_mode"),
            "planned_tools": plan.get("planned_tools"),
            "reasoning_steps": plan.get("reasoning_steps"),
            "missing_context_policy": plan.get("missing_context_policy"),
        },
        ensure_ascii=False,
        indent=2,
    )
    trace_text = json.dumps(list(trace or [])[:10], ensure_ascii=False, default=str)[:6500]
    messages: List[Dict[str, str]] = [{"role": "system", "content": _system_prompt()}]
    messages.extend(_history_to_messages(history, limit=6))
    messages.append(
        {
            "role": "user",
            "content": (
                "[Agent planner output]\n"
                f"{plan_text}\n\n"
                "[Selected tool observations]\n"
                f"{trace_text}\n\n"
                "[Full compact CXR-CAD context]\n"
                f"{context_text}\n\n"
                "[User question]\n"
                f"{question.strip()}\n\n"
                "답변 형식: 한국어 markdown. 너무 짧게 끝내지 말고, 케이스별 근거와 '현재 컨텍스트에서 부족한 점'을 명확히 구분하세요. "
                "질문이 화질/재촬영이면 품질 tool output이 없을 때 '판정 불가'와 재촬영 판단 체크리스트를 제시하세요. "
                "질문이 판독문 초안이면 Findings/Impression을 새로 간결하게 합성하되, 지원되지 않는 소견은 만들지 마세요."
            ),
        }
    )
    return messages


def _fallback_payload(
    *,
    question: str,
    compact: Mapping[str, Any],
    planned_tools: Sequence[str],
    chat_trace: Sequence[Mapping[str, Any]],
    chat_plan: Mapping[str, Any],
    error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "answer": fallback_agent_reply(question, compact),
        "engine": "local_grounded_fallback",
        "model": None,
        "fallback": True,
        "error": error,
        "used_context_tools": list(planned_tools),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "usage": {},
        "safety_note": compact.get("safety_note") or "최종 진단이 아니며 의료진 검토가 필요합니다.",
        "tool_trace": list(chat_trace),
        "agent_plan": [dict(chat_plan)],
    }


def generate_llm_agent_reply(
    *,
    question: str,
    agent_result: Mapping[str, Any],
    history: Optional[Sequence[Mapping[str, str]]] = None,
) -> Dict[str, Any]:
    """Generate a MedRAX-style follow-up answer with LLM-first planning.

    The previous implementation returned deterministic fastpath strings for
    many common questions. For demo and agentic UX, the default is now:
    planner -> context-tool trace -> LLM synthesis. Set
    CXR_AGENT_TOOL_FIRST_FASTPATH=1 only when you explicitly want the old faster
    local behavior.
    """
    question = (question or "").strip() or "이 케이스 묶음의 핵심 우선순위를 간단히 정리해줘."
    compact = compact_agent_result(agent_result)
    chat_plan = _plan_chat_context_tools(question, compact)
    planned_tools = list(chat_plan.get("planned_tools") or [])
    chat_trace = _chat_tool_trace(planned_tools, compact)

    if (not _env_bool("CXR_AGENT_LLM_FIRST", True)) and _env_bool("CXR_AGENT_TOOL_FIRST_FASTPATH", False) and _fastpath_supported(question):
        return {
            "answer": fallback_agent_reply(question, compact, include_notice=False),
            "engine": "tool_context_fastpath",
            "model": "CXR-CAD runtime tools",
            "fallback": False,
            "used_context_tools": planned_tools,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "usage": {"mode": "tool_first_fastpath"},
            "safety_note": compact.get("safety_note") or "최종 진단이 아니며 의료진 검토가 필요합니다.",
            "tool_trace": chat_trace,
            "agent_plan": [chat_plan],
        }

    if not (_env_bool("CXR_AGENT_LLM_ENABLED", True) and _llm_configured()):
        return _fallback_payload(
            question=question,
            compact=compact,
            planned_tools=planned_tools,
            chat_trace=chat_trace,
            chat_plan=chat_plan,
            error="LLM endpoint is not configured. Set OPENAI_API_KEY or CXR_AGENT_LLM_API_KEY.",
        )

    try:
        answer_messages = _build_answer_messages(question, compact, history or [], chat_plan, chat_trace)
        llm = _openai_compatible_chat(answer_messages)
        return {
            "answer": llm["answer"],
            "engine": "llm_planner_tool_synthesis",
            "model": llm.get("model"),
            "fallback": False,
            "used_context_tools": planned_tools,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "usage": llm.get("raw_usage") or {},
            "safety_note": compact.get("safety_note") or "최종 진단이 아니며 의료진 검토가 필요합니다.",
            "tool_trace": chat_trace,
            "agent_plan": [chat_plan],
        }
    except Exception as exc:
        return _fallback_payload(
            question=question,
            compact=compact,
            planned_tools=planned_tools,
            chat_trace=chat_trace,
            chat_plan=chat_plan,
            error=str(exc),
        )
