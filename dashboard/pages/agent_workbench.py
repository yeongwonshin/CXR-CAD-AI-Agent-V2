""" Agentic Case Workbench — MedRAX-style multi-image CXR workflow.

기존 CXR-CAD 모델 학습 결과를 그대로 사용하면서, 플랫폼 레벨에서
MedRAX의 강점인 multi-image 입력, 대화형 agent, DICOM 처리, 케이스 비교,
이미지별 판독문 초안/의료진 피드백을 제공하는 Streamlit 페이지입니다.
"""

from __future__ import annotations

import base64
import io
import os
from html import escape
from typing import Any, Dict, Iterable, List

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000")

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

DISEASE_LABELS_KR = {
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

MODEL_OPTIONS = {
    "ensemble": "Ensemble (Recommended)",
    "densenet": "DenseNet-121",
    "efficientnet": "EfficientNet-B4",
    "vit": "ViT-B/16",
}

FEEDBACK_TYPES = [
    "AI 판단 동의",
    "AI 판단 불일치",
    "히트맵 위치 부정확",
    "질환 라벨 수정",
    "판독의 코멘트",
]

st.set_page_config(
    page_title="CXR-CAD | Agentic Case Workbench",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 8% 8%, rgba(37,99,235,0.15), transparent 28%),
        radial-gradient(circle at 92% 14%, rgba(20,184,166,0.14), transparent 28%),
        linear-gradient(180deg, #f8fbff 0%, #eef7ff 52%, #f8fafc 100%);
}
.main .block-container { max-width: 1460px; padding-top: 1.2rem; }
[data-testid="stSidebar"] {
    background: radial-gradient(circle at top left, rgba(56,189,248,0.22), transparent 30%), linear-gradient(180deg, #08111f 0%, #0f172a 52%, #111827 100%);
}
[data-testid="stSidebar"] * { color: #e5edf8 !important; }
.agent-header {
    position: relative; overflow:hidden; border-radius: 30px; padding: 1.65rem 2rem; margin-bottom: 1.1rem;
    background: radial-gradient(circle at 83% 18%, rgba(125,211,252,0.26), transparent 25%), linear-gradient(135deg, #08111f 0%, #12345f 50%, #0f766e 100%);
    border: 1px solid rgba(125,211,252,0.26); box-shadow: 0 24px 58px rgba(15,23,42,0.20); color:white;
}
.agent-header .eyebrow { color:#a7f3d0 !important; font-size:0.72rem; letter-spacing:0.16em; text-transform:uppercase; font-weight:900; margin-bottom:0.35rem; }
.agent-header h1 { margin:0; font-size:1.85rem; font-weight:900; color:white !important; letter-spacing:-0.04em; }
.agent-header p { margin:0.45rem 0 0; font-size:0.92rem; color:#dbeafe !important; line-height:1.55; max-width: 980px; }
.agent-pill { display:inline-flex; align-items:center; gap:0.35rem; padding:0.34rem 0.72rem; border-radius:999px; margin:0.85rem 0.35rem 0 0; background:rgba(255,255,255,0.11); border:1px solid rgba(255,255,255,0.18); color:#e0f2fe !important; font-size:0.75rem; font-weight:800; }
.agent-card {
    background: rgba(255,255,255,0.94); border: 1px solid rgba(148,163,184,0.24); border-radius: 22px; padding: 1.1rem 1.2rem;
    box-shadow: 0 18px 42px rgba(15,23,42,0.08); margin-bottom: 1rem; backdrop-filter: blur(10px);
}
.agent-card h3, .agent-card h4 { margin:0 0 0.5rem; color:#0f172a !important; font-weight:900; }
.agent-card p { color:#475569 !important; font-size:0.88rem; line-height:1.55; }
.section-title { font-size:1.05rem; font-weight:900; color:#0f172a; margin:1.05rem 0 0.55rem; display:flex; align-items:center; gap:0.5rem; }
.section-title::before { content:""; width:0.56rem; height:0.56rem; border-radius:999px; background:linear-gradient(135deg,#2563eb,#14b8a6); box-shadow:0 0 0 5px rgba(14,165,233,0.12); }
.metric-card { border-radius:20px; padding:1rem; background:linear-gradient(135deg,#ffffff,#eff6ff); border:1px solid rgba(125,211,252,0.35); box-shadow:0 14px 32px rgba(15,23,42,0.08); }
.metric-card .value { font-size:1.75rem; font-weight:900; color:#0f172a !important; letter-spacing:-0.04em; }
.metric-card .label { font-size:0.72rem; color:#64748b !important; font-weight:800; letter-spacing:0.07em; text-transform:uppercase; }
.triage { border-radius:18px; padding:1rem 1.1rem; border:1px solid rgba(148,163,184,0.22); margin-bottom:0.9rem; }
.triage h4 { margin:0; font-size:1.05rem; font-weight:900; color:#0f172a !important; }
.triage p { margin:0.35rem 0 0; color:#334155 !important; line-height:1.5; }
.triage.urgent { background:linear-gradient(135deg,#ffe4e6,#fee2e2); border-color:rgba(225,29,72,0.45); }
.triage.high { background:linear-gradient(135deg,#ffedd5,#fef3c7); border-color:rgba(245,158,11,0.45); }
.triage.normal { background:linear-gradient(135deg,#ecfeff,#eff6ff); border-color:rgba(14,165,233,0.35); }
.triage.demo { background:linear-gradient(135deg,#eef2ff,#f5f3ff); border-color:rgba(124,58,237,0.35); }
.disease-tag { display:inline-flex; margin:0.18rem; padding:0.32rem 0.62rem; border-radius:999px; background:linear-gradient(135deg,#dbeafe,#ccfbf1); color:#0f172a !important; font-size:0.77rem; font-weight:850; border:1px solid rgba(14,165,233,0.22); }
.chat-panel {
    border-radius: 24px; padding: 1.05rem; min-height: 300px; max-height: 520px; overflow-y: auto;
    background: radial-gradient(circle at 18% 12%, rgba(37,99,235,0.10), transparent 30%), linear-gradient(180deg,#f8fbff,#eef6ff);
    border:1px solid rgba(125,211,252,0.34); box-shadow: inset 0 1px 0 rgba(255,255,255,0.75), 0 18px 42px rgba(15,23,42,0.07);
}
.chat-row { display:flex; width:100%; margin:0.56rem 0; }
.chat-row.user { justify-content:flex-end; }
.chat-row.agent { justify-content:flex-start; }
.chat-bubble { max-width: 86%; border-radius: 18px; padding:0.82rem 0.95rem; box-shadow:0 12px 28px rgba(15,23,42,0.10); }
.chat-bubble.user { color:white !important; background:linear-gradient(135deg,#2563eb,#0f766e); border:1px solid rgba(255,255,255,0.24); border-bottom-right-radius:6px; }
.chat-bubble.agent { color:#0f172a !important; background:rgba(255,255,255,0.96); border:1px solid rgba(148,163,184,0.30); border-bottom-left-radius:6px; }
.chat-role { font-size:0.68rem; font-weight:900; letter-spacing:0.08em; text-transform:uppercase; opacity:0.76; margin-bottom:0.35rem; }
.chat-body { font-size:0.89rem; line-height:1.58; white-space:pre-wrap; }
.chat-hint { border-radius:18px; padding:0.8rem 0.95rem; background:rgba(219,234,254,0.78); border:1px solid rgba(96,165,250,0.25); color:#1e293b !important; font-size:0.82rem; line-height:1.48; }
.small-muted { color:#64748b !important; font-size:0.78rem; line-height:1.45; }
.agent-console { border-radius:22px; padding:1rem 1.1rem; background:linear-gradient(135deg,#0f172a,#12345f); color:#e0f2fe !important; border:1px solid rgba(125,211,252,0.30); box-shadow:0 18px 42px rgba(15,23,42,0.16); margin:0.8rem 0 1rem; }
.agent-console h4 { margin:0 0 0.45rem; color:#ffffff !important; font-weight:900; }
.tool-chip { display:inline-flex; align-items:center; gap:0.25rem; padding:0.26rem 0.58rem; border-radius:999px; margin:0.12rem; background:rgba(219,234,254,0.13); border:1px solid rgba(125,211,252,0.25); color:#dbeafe !important; font-size:0.75rem; font-weight:800; }
.trace-note { color:#bae6fd !important; font-size:0.8rem; line-height:1.45; }
</style>
""",
    unsafe_allow_html=True,
)


def _guess_content_type(filename: str, fallback: str | None) -> str:
    lower = filename.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lower.endswith((".dcm", ".dicom")):
        return "application/dicom"
    return fallback or "application/octet-stream"


def call_agent_api(files, model_key: str, threshold: float, question: str) -> dict | None:
    multipart = []
    for f in files:
        raw = f.getvalue()
        multipart.append(("files", (f.name, raw, _guess_content_type(f.name, getattr(f, "type", None)))))
    try:
        resp = requests.post(
            f"{API_URL}/agent/analyze",
            params={"model": model_key, "threshold": threshold, "question": question},
            files=multipart,
            timeout=240,
        )
        if resp.status_code == 200:
            return resp.json()
        st.error(f"Agent API 오류 {resp.status_code}: {resp.text}")
    except requests.exceptions.ConnectionError:
        st.error("백엔드 API에 연결할 수 없습니다. `uvicorn api.main:app --reload --port 8000`을 먼저 실행하세요.")
    except Exception as exc:
        st.error(f"Agent 분석 요청 중 오류가 발생했습니다: {exc}")
    return None


def _short_text(value: Any, limit: int = 700) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def compact_chat_result_payload(result: dict) -> dict:
    """Strip image/base64-heavy fields before /agent/chat.

    The Workbench keeps the full result for image tabs and Grad-CAM rendering,
    but chat only needs compact tool outputs.  This removes GradCAM_Base64 and
    verbose report dictionaries, which makes follow-up questions feel much more
    like MedRAX's lightweight thread interaction.
    """
    result = result or {}
    compact_cases = []
    for idx, case in enumerate(result.get("cases", []) or []):
        prediction = case.get("prediction", {}) or {}
        profile = case.get("agent_profile", {}) or {}
        quality = profile.get("quality_check", {}) or prediction.get("Quality_Check", {}) or {}
        triage = profile.get("triage_assessment", {}) or prediction.get("Triage_Assessment", {}) or {}
        anatomy = profile.get("anatomy_assessment", {}) or prediction.get("Anatomy_Assessment", {}) or {}
        rois = []
        for roi in (anatomy.get("focus_rois") or [])[:4]:
            rois.append({
                "label_kr": roi.get("label_kr") or roi.get("label"),
                "priority_score": roi.get("priority_score"),
                "related_findings": (roi.get("related_findings") or [])[:4],
                "review_hint": _short_text(roi.get("review_hint"), 160),
            })
        probs = case.get("probabilities", {}) or {}
        top_probabilities = [
            {"label": label, "label_kr": DISEASE_LABELS_KR.get(label, label), "probability": float(prob)}
            for label, prob in sorted(probs.items(), key=lambda item: float(item[1] or 0.0), reverse=True)[:5]
        ]
        compact_cases.append({
            "index": idx + 1,
            "filename": case.get("filename"),
            "case_id": case.get("case_id"),
            "top_disease": case.get("top_disease"),
            "top_disease_kr": DISEASE_LABELS_KR.get(str(case.get("top_disease")), str(case.get("top_disease"))),
            "top_probability": case.get("top_probability"),
            "detected_diseases": (case.get("detected_diseases") or [])[:8],
            "top_probabilities": top_probabilities,
            "probabilities": {label: round(float(prob or 0.0), 4) for label, prob in probs.items()},
            "is_placeholder": case.get("is_placeholder"),
            "quality_check": {
                "quality_grade": quality.get("quality_grade"),
                "quality_score": quality.get("quality_score"),
                "flags": (quality.get("flags") or [])[:3],
            },
            "triage_assessment": {
                "triage_level": triage.get("triage_level"),
                "triage_label_kr": triage.get("triage_label_kr"),
                "reason": _short_text(triage.get("reason"), 240),
            },
            "anatomy_assessment": {
                "recommended_review_order": (anatomy.get("recommended_review_order") or [])[:5],
                "focus_rois": rois,
                "disclaimer": _short_text(anatomy.get("disclaimer"), 220),
            },
            "report_draft": _short_text(case.get("report_draft") or prediction.get("Report_Draft"), 700),
            "findings_kr": _short_text(prediction.get("Findings_KR"), 320),
            "impression_kr": _short_text(prediction.get("Impression_KR"), 320),
            "has_gradcam": bool(prediction.get("GradCAM_Base64") and len(str(prediction.get("GradCAM_Base64"))) > 500),
        })

    summary = result.get("agent_summary", {}) or {}
    comparison = summary.get("comparison", {}) or {}
    return {
        "status": result.get("status"),
        "model_key": result.get("model_key"),
        "threshold": result.get("threshold"),
        "case_count": result.get("case_count") or len(compact_cases),
        "cases": compact_cases,
        "agent_summary": {
            "narrative": _short_text(summary.get("narrative"), 700),
            "placeholder_count": summary.get("placeholder_count"),
            "comparison": {
                "enabled": comparison.get("enabled", False),
                "summary": _short_text(comparison.get("summary"), 500),
                "probability_deltas": (comparison.get("probability_deltas") or [])[:6],
            },
            "safety_note": _short_text(summary.get("safety_note"), 280),
        },
        "safety_note": result.get("safety_note"),
    }


def _is_compact_chat_payload(result: dict) -> bool:
    """True when the payload is already stripped for /agent/chat.

    Re-compacting an already compact payload drops quality/triage/report fields
    because there is no nested `prediction` or `agent_profile` left. That was
    the main reason follow-up answers showed 품질=None or Findings=-.
    """
    cases = (result or {}).get("cases") or []
    if not isinstance(cases, list) or not cases:
        return False
    first = cases[0] if isinstance(cases[0], dict) else {}
    return "prediction" not in first and any(k in first for k in ["quality_check", "triage_assessment", "report_draft", "top_probabilities"])


def call_agent_chat_api(question: str, result: dict, history: List[Dict[str, str]]) -> dict | None:
    """Send a follow-up question to the backend LLM agent.

    The UI keeps the same chat bubbles/input, while the answer engine moves to
    the MedRAX-like backend path: LLM brain over existing CXR-CAD tool outputs.
    """
    chat_context = result if _is_compact_chat_payload(result) else compact_chat_result_payload(result)
    payload = {
        "question": question.strip(),
        "result": chat_context,
        "history": history[-6:],
    }
    try:
        resp = requests.post(f"{API_URL}/agent/chat", json=payload, timeout=140)
        if resp.status_code == 200:
            return resp.json()
        st.error(f"Agent Chat API 오류 {resp.status_code}: {resp.text}")
    except requests.exceptions.ConnectionError:
        st.error("백엔드 API에 연결할 수 없습니다. `uvicorn api.main:app --reload --port 8000`을 먼저 실행하세요.")
    except Exception as exc:
        st.error(f"Agent 후속 질의 중 오류가 발생했습니다: {exc}")
    return None


def call_feedback_api(payload: dict) -> dict | None:
    try:
        resp = requests.post(f"{API_URL}/feedback", json=payload, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        st.error(f"피드백 저장 오류 {resp.status_code}: {resp.text}")
    except requests.exceptions.ConnectionError:
        st.error("백엔드 API에 연결할 수 없어 피드백을 저장하지 못했습니다.")
    return None


def check_api_health() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

def check_agent_status() -> dict | None:
    """Read backend LLM-agent runtime mode without exposing secrets."""
    try:
        r = requests.get(f"{API_URL}/agent/status", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None



def get_gradcam_risk_color(prob: float, threshold: float) -> str:
    """Use the same red/yellow/teal risk palette as the main App Grad-CAM chart.

    The Agent Workbench keeps the same prediction values as before; this only
    changes the visual encoding so high-probability bars visually match the
    Grad-CAM risk language already used on the main upload page.
    """
    if prob >= 0.5:
        return "#e11d48"  # Grad-CAM-like hot red: high attention / high risk
    if prob >= threshold:
        return "#f59e0b"  # warning yellow/orange: above operating threshold
    return "#14b8a6"      # teal/green: below threshold


def render_bar_chart(probs: Dict[str, float], threshold: float) -> go.Figure:
    items = sorted(probs.items(), key=lambda item: item[1])
    labels = [DISEASE_LABELS_KR.get(k, k) for k, _ in items]
    values = [v for _, v in items]
    colors = [get_gradcam_risk_color(v, threshold) for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels,
        x=values,
        orientation="h",
        marker=dict(color=colors),
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
        textfont=dict(size=11, family="Inter", color="#475569"),
        hovertemplate="<b>%{y}</b><br>확률: %{x:.2%}<extra></extra>",
    ))
    fig.add_vline(
        x=threshold,
        line=dict(color="#f59e0b", width=2, dash="dash"),
        annotation=dict(
            text=f"임계값 ({threshold:.0%})",
            font=dict(size=10, color="#b45309"),
            yref="paper",
            y=1.05,
        ),
    )
    fig.update_layout(
        height=480,
        margin=dict(l=0, r=45, t=30, b=20),
        xaxis=dict(
            range=[0, 1.12],
            tickformat=".0%",
            showgrid=True,
            gridcolor="rgba(148,163,184,0.18)",
            tickfont=dict(size=10, family="Inter", color="#64748b"),
        ),
        yaxis=dict(tickfont=dict(size=12, family="Inter", color="#334155")),
        plot_bgcolor="rgba(255,255,255,0)",
        paper_bgcolor="rgba(255,255,255,0)",
        font=dict(family="Inter"),
    )
    return fig


def prediction_summary_payload(case: dict, edited_report: str) -> dict:
    prediction = case.get("prediction", {}) or {}
    return {
        "filename": case.get("filename"),
        "case_id": case.get("case_id"),
        "top_disease": case.get("top_disease"),
        "top_probability": case.get("top_probability"),
        "detected_diseases": case.get("detected_diseases", []),
        "probabilities": case.get("probabilities", {}),
        "model_used": prediction.get("Model_Used"),
        "model_key": prediction.get("Model_Key"),
        "is_placeholder": case.get("is_placeholder"),
        "report_draft_kr": edited_report,
        "quality_check": (case.get("agent_profile") or {}).get("quality_check", {}),
        "triage_assessment": (case.get("agent_profile") or {}).get("triage_assessment", {}),
        "anatomy_assessment": (case.get("agent_profile") or {}).get("anatomy_assessment", {}),
    }


def submit_feedback(case: dict, feedback_type: str, threshold: float, corrected_labels: list[str], comment: str, reviewer_id: str, edited_report: str) -> None:
    payload = {
        "case_id": case.get("case_id", "CXR-UNKNOWN"),
        "feedback_type": feedback_type,
        "original_top_disease": case.get("top_disease"),
        "corrected_labels": corrected_labels,
        "comment": comment.strip(),
        "reviewer_id": reviewer_id.strip() or None,
        "model_key": (case.get("prediction", {}) or {}).get("Model_Key"),
        "threshold": threshold,
        "prediction_summary": prediction_summary_payload(case, edited_report),
    }
    saved = call_feedback_api(payload)
    if saved:
        st.success(f"{saved.get('message', '피드백이 저장되었습니다')} 큐 ID: {saved.get('queue_id', '-')}")


def safe_preview_image(uploaded_lookup: Dict[str, bytes], filename: str) -> Image.Image | None:
    raw = uploaded_lookup.get(filename)
    if raw is None or filename.lower().endswith((".dcm", ".dicom")):
        return None
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except Exception:
        return "-"


def _label_kr(label: str | None) -> str:
    if not label:
        return "-"
    return DISEASE_LABELS_KR.get(str(label), str(label).replace("_", " "))


def _case_name(case: dict, idx: int) -> str:
    filename = case.get("filename") or f"case_{idx + 1}"
    return f"{idx + 1}번 영상({filename})"


def _case_display_label(case: dict, idx: int, max_len: int = 26) -> str:
    filename = str(case.get("filename") or f"case_{idx + 1}")
    if len(filename) > max_len:
        filename = filename[: max_len - 3].rstrip() + "..."
    return f"{idx + 1}. {filename}"


def _has_gradcam(case: dict) -> bool:
    prediction = case.get("prediction", {}) or {}
    gradcam = prediction.get("GradCAM_Base64", "")
    return bool(gradcam and len(str(gradcam)) > 500)


def build_all_case_overview_df(cases: List[dict], threshold_value: float) -> pd.DataFrame:
    """Build one row per uploaded image so the Workbench compares all cases at once."""
    rows: List[dict] = []
    for idx, case in enumerate(cases or []):
        profile = case.get("agent_profile", {}) or {}
        quality = profile.get("quality_check", {}) or {}
        triage = profile.get("triage_assessment", {}) or {}
        probs = case.get("probabilities", {}) or {}
        top3 = sorted(probs.items(), key=lambda item: float(item[1] or 0.0), reverse=True)[:3]
        detected = case.get("detected_diseases", []) or []
        rows.append({
            "영상": idx + 1,
            "파일명": case.get("filename", "-"),
            "Top 소견": _label_kr(case.get("top_disease")),
            "Top 확률": _pct(case.get("top_probability")),
            "임계값 이상 소견": ", ".join(_label_kr(d) for d in detected) if detected else "없음",
            "상위 3개 확률": " / ".join(f"{_label_kr(label)} {_pct(prob)}" for label, prob in top3),
            "우선도": triage.get("triage_label_kr", "-"),
            "품질": f"{quality.get('quality_grade', '-')} ({quality.get('quality_score', '-')}/100)",
            "Grad-CAM": "제공" if _has_gradcam(case) else "없음/Placeholder",
            "DICOM": "예" if ((case.get("prediction", {}) or {}).get("Image_Metadata", {}) or {}).get("is_dicom_input") else "아니오",
            "Placeholder": "예" if case.get("is_placeholder") else "아니오",
        })
    return pd.DataFrame(rows)


def build_all_case_probability_df(cases: List[dict]) -> pd.DataFrame:
    """Return a wide probability table: each image row contains every disease probability."""
    rows: List[dict] = []
    for idx, case in enumerate(cases or []):
        probs = case.get("probabilities", {}) or {}
        row = {
            "영상": idx + 1,
            "파일명": case.get("filename", "-"),
            "Top 소견": _label_kr(case.get("top_disease")),
        }
        for label in DISEASE_LABELS:
            row[DISEASE_LABELS_KR.get(label, label)] = _pct(probs.get(label, 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def render_all_case_probability_heatmap(cases: List[dict], threshold_value: float) -> go.Figure | None:
    """Render all uploaded images x disease probabilities as a simultaneous comparison heatmap."""
    if not cases:
        return None
    labels = DISEASE_LABELS
    x_labels = [_case_display_label(case, idx) for idx, case in enumerate(cases)]
    y_labels = [DISEASE_LABELS_KR.get(label, label) for label in labels]
    z = []
    hover = []
    for label in labels:
        row_vals = []
        row_hover = []
        for idx, case in enumerate(cases):
            probs = case.get("probabilities", {}) or {}
            value = float(probs.get(label, 0.0) or 0.0)
            row_vals.append(value)
            row_hover.append(
                f"영상: {_case_display_label(case, idx)}<br>질환: {DISEASE_LABELS_KR.get(label, label)}<br>확률: {value:.1%}"
            )
        z.append(row_vals)
        hover.append(row_hover)

    threshold_value = max(0.01, min(float(threshold_value or 0.3), 0.99))
    colorscale = [
        [0.0, "#14b8a6"],
        [max(0.01, threshold_value), "#f59e0b"],
        [1.0, "#e11d48"],
    ]
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        zmin=0,
        zmax=1,
        colorscale=colorscale,
        colorbar=dict(title="확률", tickformat=".0%"),
        text=[[f"{v:.0%}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
        hoverinfo="text",
        hovertext=hover,
    ))
    fig.update_layout(
        height=max(430, 28 * len(labels) + 150),
        margin=dict(l=0, r=0, t=30, b=70),
        xaxis=dict(tickangle=-25, tickfont=dict(size=10, family="Inter", color="#334155")),
        yaxis=dict(tickfont=dict(size=11, family="Inter", color="#334155")),
        plot_bgcolor="rgba(255,255,255,0)",
        paper_bgcolor="rgba(255,255,255,0)",
        font=dict(family="Inter"),
    )
    return fig


def _top_case_lines(result: dict, limit: int = 4) -> List[str]:
    cases = result.get("cases", []) or []
    ranked = sorted(
        enumerate(cases),
        key=lambda item: float(item[1].get("top_probability") or 0.0),
        reverse=True,
    )
    lines: List[str] = []
    for idx, case in ranked[:limit]:
        triage = ((case.get("agent_profile") or {}).get("triage_assessment") or {})
        lines.append(
            f"- {_case_name(case, idx)}: {_label_kr(case.get('top_disease'))} {_pct(case.get('top_probability'))} · "
            f"{triage.get('triage_label_kr', '우선도 확인 필요')}"
        )
    return lines


def _mentioned_diseases(question: str) -> List[str]:
    q = question.lower()
    found: List[str] = []
    for label, label_kr in DISEASE_LABELS_KR.items():
        if label.lower() in q or label_kr.lower() in q:
            found.append(label)
    return found


def _disease_specific_reply(result: dict, labels: List[str]) -> str:
    cases = result.get("cases", []) or []
    if not cases:
        return "아직 분석된 영상이 없어 질환별 답변을 만들 수 없습니다."
    blocks = ["질문에서 언급된 소견을 영상별로 확인했습니다."]
    for label in labels:
        rows = []
        for idx, case in enumerate(cases):
            probs = case.get("probabilities", {}) or {}
            rows.append((idx, case, float(probs.get(label, 0.0))))
        rows.sort(key=lambda item: item[2], reverse=True)
        blocks.append(f"\n{_label_kr(label)} 기준:")
        for idx, case, prob in rows[:5]:
            blocks.append(f"- {_case_name(case, idx)}: {prob:.1%}")
    blocks.append("\n확률이 높게 나온 영상은 Grad-CAM과 이미지별 판독문 초안을 함께 확인하고, 최종 판단은 의료진 검토로 확정하는 흐름이 안전합니다.")
    return "\n".join(blocks)


def _comparison_reply(result: dict) -> str:
    summary = result.get("agent_summary", {}) or {}
    comparison = summary.get("comparison") or {}
    if not comparison.get("enabled"):
        return "현재 분석 묶음에는 비교 가능한 영상이 2장 이상 없거나 비교 결과가 생성되지 않았습니다. 여러 장을 업로드하면 첫 영상과 마지막 영상의 확률 변화량을 기준으로 악화·호전 후보를 정리합니다."
    lines = [comparison.get("summary", "영상 간 변화 비교 결과입니다.")]
    deltas = comparison.get("probability_deltas", []) or []
    if deltas:
        lines.append("\n변화량이 큰 항목:")
        for item in deltas[:6]:
            label = item.get("label_kr") or _label_kr(item.get("label"))
            first_p = _pct(item.get("first_probability"))
            last_p = _pct(item.get("last_probability"))
            delta = float(item.get("delta") or 0.0)
            direction = "증가" if delta > 0 else "감소" if delta < 0 else "변화 없음"
            lines.append(f"- {label}: {first_p} → {last_p} ({direction} {abs(delta):.1%}p)")
    lines.append("\n동일 환자의 시간축 비교라면 촬영 조건과 자세 차이도 함께 확인해야 합니다.")
    return "\n".join(lines)


def _triage_reply(result: dict) -> str:
    lines = ["우선 검토가 필요한 영상부터 정리했습니다."]
    case_lines = _top_case_lines(result, limit=6)
    lines.extend(case_lines or ["- 분석된 케이스가 없습니다."])
    lines.append("\n특히 기흉, 폐부종, 폐렴, 흉수, 심비대 관련 확률이 높거나 Grad-CAM 위치가 임상 소견과 맞는 경우 우선 검토 대상으로 두는 것이 좋습니다.")
    return "\n".join(lines)


def _quality_reply(result: dict) -> str:
    cases = result.get("cases", []) or []
    lines = ["영상 품질 점검 결과입니다."]
    for idx, case in enumerate(cases):
        quality = (((case.get("agent_profile") or {}).get("quality_check")) or {})
        flags = quality.get("flags") or []
        flag_text = "; ".join(str(x) for x in flags[:3]) if flags else "특이 품질 경고 없음"
        lines.append(
            f"- {_case_name(case, idx)}: {quality.get('quality_grade', '-')} · "
            f"{quality.get('quality_score', '-')}/100 · {flag_text}"
        )
    lines.append("\n품질 등급이 낮은 영상은 모델 확률과 Grad-CAM을 단독 근거로 쓰지 말고 재촬영 여부 또는 판독 보수성을 함께 검토하세요.")
    return "\n".join(lines)


def _roi_reply(result: dict) -> str:
    cases = result.get("cases", []) or []
    lines = ["해부학 ROI 검토 포인트입니다. 이 ROI는 새 segmentation 학습 결과가 아니라 기존 예측과 표준 해부학 위치를 연결한 검토용 스캐폴드입니다."]
    for idx, case in enumerate(cases):
        anatomy = (((case.get("agent_profile") or {}).get("anatomy_assessment")) or {})
        rois = anatomy.get("focus_rois") or []
        if not rois:
            lines.append(f"- {_case_name(case, idx)}: ROI 후보 없음")
            continue
        roi_text = ", ".join(f"{r.get('label_kr')}({float(r.get('priority_score', 0.0)):.0%})" for r in rois[:4])
        lines.append(f"- {_case_name(case, idx)}: {roi_text}")
    return "\n".join(lines)


def _report_reply(result: dict) -> str:
    cases = result.get("cases", []) or []
    lines = ["이미지별 판독문 초안은 아래 탭의 편집창에서 각각 수정·다운로드할 수 있습니다. 핵심 초안 요약은 다음과 같습니다."]
    for idx, case in enumerate(cases[:4]):
        draft = str(case.get("report_draft") or "").strip().replace("\n", " ")
        if len(draft) > 260:
            draft = draft[:260].rstrip() + "..."
        lines.append(f"\n{_case_name(case, idx)}\n- {draft or '초안 없음'}")
    return "\n".join(lines)


def _gradcam_reply(result: dict) -> str:
    cases = result.get("cases", []) or []
    lines = ["Grad-CAM 표시 가능 여부입니다."]
    for idx, case in enumerate(cases):
        pred = case.get("prediction", {}) or {}
        has_cam = bool(pred.get("GradCAM_Base64") and len(str(pred.get("GradCAM_Base64"))) > 500)
        lines.append(f"- {_case_name(case, idx)}: {'Grad-CAM 표시 가능' if has_cam else 'Grad-CAM 없음 또는 Placeholder 응답'}")
    lines.append("\nGrad-CAM은 기존 CXR-CAD 기능을 그대로 사용하며, 이미지별 탭의 원본 이미지 아래에서 확인할 수 있습니다.")
    return "\n".join(lines)


def build_agent_reply(question: str, result: dict) -> str:
    question = (question or "").strip()
    q = question.lower()
    mentioned = _mentioned_diseases(question)
    if mentioned:
        return _disease_specific_reply(result, mentioned)
    if any(k in q for k in ["비교", "변화", "악화", "호전", "이전", "마지막", "첫", "compare", "change", "worse", "better"]):
        return _comparison_reply(result)
    if any(k in q for k in ["품질", "화질", "선명", "흐림", "quality", "재촬영"]):
        return _quality_reply(result)
    if any(k in q for k in ["roi", "해부학", "위치", "폐야", "심장", "종격동", "늑골", "어디", "location"]):
        return _roi_reply(result)
    if any(k in q for k in ["grad", "cam", "히트맵", "heatmap", "근거"]):
        return _gradcam_reply(result)
    if any(k in q for k in ["판독", "초안", "소견", "리포트", "보고서", "report", "draft"]):
        return _report_reply(result)
    if any(k in q for k in ["우선", "응급", "위험", "급", "triage", "먼저", "priority"]):
        return _triage_reply(result)

    summary = result.get("agent_summary", {}) or {}
    lines = [summary.get("narrative") or "분석 결과를 기준으로 답변합니다."]
    lines.append("\n핵심 우선순위:")
    lines.extend(_top_case_lines(result, limit=4) or ["- 분석된 케이스가 없습니다."])
    lines.append("\n추가로 '비교해줘', '폐렴만 정리해줘', '품질 문제 있어?', 'Grad-CAM 근거 보여줘', '판독문 초안 요약해줘'처럼 이어서 물어볼 수 있습니다.")
    return "\n".join(lines)


def build_initial_chat(result: dict, initial_question: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if initial_question.strip():
        messages.append({"role": "user", "content": initial_question.strip()})
        messages.append({"role": "agent", "content": build_agent_reply(initial_question, result)})
    else:
        messages.append({
            "role": "agent",
            "content": "분석이 완료되었습니다. 이 케이스 묶음에 대해 계속 질문할 수 있습니다. 예: '가장 위험한 영상부터 정리해줘', '첫 번째와 마지막 영상을 비교해줘', '심비대 확률만 알려줘', '판독문 초안을 요약해줘'.",
        })
    return messages


def render_chat_messages(messages: List[Dict[str, str]]) -> None:
    html_parts = ['<div class="chat-panel">']
    if not messages:
        html_parts.append('<div class="chat-hint">분석 실행 후 Agent와 후속 질문을 주고받을 수 있습니다.</div>')
    for msg in messages:
        role = msg.get("role", "agent")
        is_user = role == "user"
        role_label = "YOU" if is_user else "CXR AGENT"
        body = escape(str(msg.get("content", ""))).replace("\n", "<br>")
        klass = "user" if is_user else "agent"
        html_parts.append(
            f'<div class="chat-row {klass}"><div class="chat-bubble {klass}">'
            f'<div class="chat-role">{role_label}</div><div class="chat-body">{body}</div>'
            f'</div></div>'
        )
    html_parts.append('</div>')
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def _tool_chips(tool_names: List[str]) -> str:
    if not tool_names:
        return "<span class='tool-chip'>No tool trace</span>"
    return " ".join(f"<span class='tool-chip'>✓ {escape(str(name))}</span>" for name in tool_names)


def _latest_agent_meta(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    for msg in reversed(messages or []):
        if msg.get("role") in {"agent", "assistant"} and isinstance(msg.get("meta"), dict):
            return msg.get("meta") or {}
    return {}


def render_latest_agent_trace(messages: List[Dict[str, Any]]) -> None:
    """Show the MedRAX-like planner/tool trace for the latest chat answer."""
    meta = _latest_agent_meta(messages)
    if not meta:
        return

    used_tools = list(meta.get("used_context_tools") or [])
    engine = meta.get("engine", "-")
    model = meta.get("model") or "local fallback"
    fallback = "YES" if meta.get("fallback") else "NO"
 

    with st.expander("Planner / Tool trace 자세히 보기", expanded=False):
        plan = meta.get("agent_plan") or []
        if plan:
            plan0 = plan[0] if isinstance(plan, list) else plan
            st.markdown("**Planner reasoning**")
            for step in (plan0.get("reasoning_steps") or [])[:8]:
                st.caption(f"• {step}")
            st.json(plan0, expanded=False)
        trace = meta.get("tool_trace") or []
        if trace:
            rows = []
            for item in trace:
                rows.append({
                    "iteration": item.get("iteration"),
                    "tool": item.get("name"),
                    "scope": item.get("scope"),
                    "status": item.get("status"),
                    "rationale": _short_text(item.get("rationale"), 180),
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        if meta.get("error"):
            st.warning(f"LLM fallback reason: {meta.get('error')}")


def render_agent_execution_console(result: dict) -> None:
    """Expose the analysis-time dynamic agent trace for demo visibility."""
    trace = list(result.get("tool_trace") or [])
    plan = list(result.get("agent_plan") or [])
    tool_names: List[str] = []
    for item in trace:
        name = item.get("name")
        if name and name not in tool_names:
            tool_names.append(str(name))

    st.markdown(
        f"""
<div class="agent-console">
  <h4>🔁 Dynamic Agent Flow</h4>
  <div class="trace-note">초기 분석에서 실제 선택·실행된 tool sequence입니다. 고정 파이프라인이 아니라 planner가 질문/상태에 따라 tool을 고릅니다.</div>
  <div style="margin-top:0.55rem;">{_tool_chips(tool_names)}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    with st.expander("초기 분석 planner / tool 실행 로그", expanded=False):
        if trace:
            rows = []
            for item in trace:
                rows.append({
                    "scope": item.get("scope", "case"),
                    "file": item.get("filename", "batch"),
                    "iteration": item.get("iteration"),
                    "tool": item.get("name"),
                    "status": item.get("status"),
                    "duration_ms": item.get("duration_ms"),
                    "rationale": _short_text(item.get("rationale"), 220),
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.info("tool_trace가 비어 있습니다. 백엔드가 최신 agentic 버전인지 확인하세요.")
        if plan:
            st.markdown("**Raw agent plan**")
            st.json(plan[:10], expanded=False)


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def submit_agent_followup(followup: str, result: dict) -> None:
    """Send a follow-up through the backend LLM agent and append UI metadata.

    The UI intentionally avoids silently substituting the old Streamlit
    hard-coded answer templates. If the backend fails, the user sees that the
    MedRAX-style agent path failed instead of a deceptively fluent fallback.
    """
    followup = (followup or "").strip()
    if not followup:
        st.warning("질문을 입력한 뒤 전송하세요.")
        return

    user_message = {"role": "user", "content": followup}
    prior_history = list(st.session_state.get("agent_chat_messages", []))
    st.session_state["agent_chat_messages"].append(user_message)
    with st.spinner("LLM Agent가 planner → tool context → answer synthesis 순서로 답변 중입니다..."):
        chat_context = st.session_state.get("agent_chat_context") or compact_chat_result_payload(result)
        chat_response = call_agent_chat_api(followup, chat_context, prior_history + [user_message])

    if chat_response and chat_response.get("answer"):
        answer = str(chat_response.get("answer"))
        agent_meta: Dict[str, Any] = chat_response
        if chat_response.get("fallback"):
            answer += (
                "\n\n⚠️ 현재 답변은 LLM 합성이 아니라 local grounded fallback입니다. "
                "대회 시연에서는 백엔드 환경변수 `OPENAI_API_KEY` 또는 `CXR_AGENT_LLM_API_KEY`를 설정해야 "
                "planner + LLM synthesis가 실제로 활성화됩니다."
            )
            if chat_response.get("error"):
                answer += f"\nFallback reason: {chat_response.get('error')}"
    else:
        answer = (
            "백엔드 MedRAX-style Agent 응답을 받지 못했습니다. "
            "FastAPI 로그와 `/agent/status`의 LLM 설정 상태를 확인하세요. "
            "이 화면은 더 이상 예전 하드코딩 답변으로 조용히 대체하지 않습니다."
        )
        agent_meta = {"engine": "backend_agent_unavailable", "fallback": True, "used_context_tools": [], "error": "No /agent/chat answer"}

    st.session_state["agent_chat_messages"].append({"role": "agent", "content": answer, "meta": agent_meta})
    st.session_state["agent_chat_nonce"] += 1
    _rerun()


def _install_enter_submit_hotkey() -> None:
    """Submit the existing follow-up textarea with Enter, without changing UI."""
    components.html(
        """
<script>
(function() {
  function bindCxrAgentEnterSubmit() {
    try {
      const doc = window.parent.document;
      const textareas = Array.from(doc.querySelectorAll('textarea'));
      const target = textareas.find((el) => {
        const label = el.getAttribute('aria-label') || '';
        const placeholder = el.getAttribute('placeholder') || '';
        return label.includes('Agent에게 후속 질문') || placeholder.includes('폐렴 의심 영상');
      });
      if (!target || target.dataset.cxrAgentEnterSubmit === '1') return;
      target.dataset.cxrAgentEnterSubmit = '1';
      target.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.metaKey && !e.altKey) {
          e.preventDefault();
          const buttons = Array.from(doc.querySelectorAll('button'));
          const sendButton = buttons.find((btn) => (btn.innerText || '').includes('질문 보내기'));
          if (sendButton) sendButton.click();
        }
      }, true);
    } catch (err) {
      // Streamlit component sandbox may block parent access on some deployments.
      // The visible submit button remains available as the fallback path.
    }
  }
  bindCxrAgentEnterSubmit();
  window.setInterval(bindCxrAgentEnterSubmit, 800);
})();
</script>
        """,
        height=0,
    )


def render_agent_chat(result: dict) -> None:
    st.markdown('<div class="section-title">Agent 대화 창</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="agent-card">
    <h3>AI Agent</h3>
    <p>업로드한 여러 영상의 분석 결과를 기존 CXR-CAD 도구 출력으로 묶고, 백엔드 LLM Agent가 그 컨텍스트 위에서 계속 답변합니다. 대화 UI와 이미지별 판독·피드백 기능은 그대로 유지됩니다.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.session_state.setdefault("agent_chat_messages", [])
    st.session_state.setdefault("agent_chat_nonce", 0)
    render_chat_messages(st.session_state["agent_chat_messages"])
    render_latest_agent_trace(st.session_state["agent_chat_messages"])

    st.markdown("**시연용 질문 예시**: `화질과 재촬영 필요성만 봐줘` · `Grad-CAM 근거와 의심 부위를 설명해줘` · `세 장 중 가장 먼저 볼 영상을 골라줘` · `판독문 초안을 Findings/Impression으로 다시 써줘`")

    st.markdown("**One-click demo prompts**")
    qcols = st.columns(4)
    demo_prompts = [
        "화질과 재촬영 필요성만 봐줘. 케이스별로 판정 가능/불가능을 구분해줘",
        "Grad-CAM 근거와 의심 부위를 케이스별로 설명해줘",
        "가장 위험한 영상부터 우선순위를 매기고 이유를 설명해줘",
        "판독문 초안을 Findings / Impression 형식으로 새로 써줘",
    ]
    for col, prompt in zip(qcols, demo_prompts):
        with col:
            if st.button(prompt.split(".")[0], key=f"demo_prompt_{abs(hash(prompt))}", use_container_width=True):
                submit_agent_followup(prompt, result)

    prefill = st.session_state.pop("agent_followup_prefill", "") if "agent_followup_prefill" in st.session_state else ""
    with st.form("agent_followup_form", clear_on_submit=False):
        input_key = f"agent_followup_input_{st.session_state['agent_chat_nonce']}"
        followup = st.text_area(
            "Agent에게 후속 질문",
            value=prefill,
            placeholder="예: 폐렴 의심 영상만 정리해줘. / 첫 번째와 마지막 영상을 비교해줘. / 어떤 영상을 먼저 봐야 해? / 품질 문제 있는 파일 있어?",
            height=135,
            key=input_key,
        )
        col_send, col_clear = st.columns([3, 1])
        with col_send:
            send = st.form_submit_button("질문 보내기", type="primary", use_container_width=True)
        with col_clear:
            clear = st.form_submit_button("대화 초기화", use_container_width=True)

    _install_enter_submit_hotkey()

    if clear:
        st.session_state["agent_chat_messages"] = build_initial_chat(result, "")
        st.session_state["agent_chat_nonce"] += 1
        _rerun()
    if send:
        submit_agent_followup(followup, result)


with st.sidebar:
    st.markdown("### Agent Workbench")
    st.caption("MedRAX식 multi-tool CXR 워크플로우")
    health = check_api_health()
    if health:
        loaded = health.get("loaded_models", [])
        st.success(f"FastAPI 연결 · 로드 모델 {len(loaded)}개")
        agent_status = check_agent_status() or {}
        if agent_status.get("llm_configured") and agent_status.get("llm_enabled"):
            st.success(f"LLM Agent ON · {agent_status.get('model', '-')}")
        else:
            st.warning("LLM Agent OFF · local fallback")
            st.caption("대회 시연 전 FastAPI 환경에 OPENAI_API_KEY 또는 CXR_AGENT_LLM_API_KEY를 설정하세요.")
    else:
        st.error("FastAPI 미연결")
    st.divider()
    model_label = st.radio("분석 모델", list(MODEL_OPTIONS.values()), index=0)
    model_key = [k for k, v in MODEL_OPTIONS.items() if v == model_label][0]
    threshold = st.slider("감지 임계값", 0.1, 0.9, 0.3, 0.05)
 
st.markdown(
    """
<div class="agent-header">
    <div class="eyebrow">MedRAX-inspired Runtime Agent</div>
    <h1>CXR-CAD - Agent Workbench</h1>
    <p>
        기존 CXR-CAD의 학습 완료 모델과 Grad-CAM, 판독문 초안, 의료진 피드백 큐는 그대로 유지하면서,
        여러 장의 X-ray/DICOM을 한 번에 입력하고 이미지별 결과·검진 초안·피드백·품질 점검·해부학 ROI·비교 요약을 제공하는 agent workbench입니다.
    </p>
    <span class="agent-pill">Multi-image upload</span>
    <span class="agent-pill">DICOM-aware routing</span>
    <span class="agent-pill">LLM-first planner</span>
    <span class="agent-pill">Tool trace console</span>
    <span class="agent-pill">Interactive chat</span>
    <span class="agent-pill">Per-case draft & feedback</span>
    <span class="agent-pill">Follow-up comparison</span>
</div>
""",
    unsafe_allow_html=True,
)

with st.expander("이 페이지가 기존 App과 다른 점", expanded=True):
    st.markdown(
        """
- **기존 App**: 1장 중심 분석, Grad-CAM, 판독문 초안, 의료진 피드백을 안정적으로 유지합니다.
- **Agent Workbench**: 여러 장을 한 케이스 묶음으로 분석하고, MedRAX처럼 Agent와 후속 질문을 계속 주고받으며, 이미지별 결과와 전체 비교를 함께 제공합니다.
- **학습 과정 변경 없음**: 새 모델을 학습하지 않고 기존 `/predict`, Grad-CAM, report draft, feedback queue를 Agent 오케스트레이션으로 확장합니다.
        """
    )

uploaded_files = st.file_uploader(
    "X-ray 또는 DICOM 파일을 여러 장 업로드하세요",
    type=["png", "jpg", "jpeg", "dcm", "dicom"],
    accept_multiple_files=True,
    help="동일 환자의 과거/현재 영상 또는 여러 케이스를 한 번에 넣어 이미지별 초안과 비교 요약을 생성합니다.",
)
question = st.text_input(
    "초기 Agent 목표 / 시연 질문",
    value="",
    placeholder="예: 화질과 재촬영 필요성을 우선 평가해줘 / Grad-CAM 근거 중심으로 설명해줘 / 가장 위험한 영상부터 정리해줘",
    help="비워두면 기본 full workup을 실행합니다. 입력하면 MedRAX식 planner가 질문 의도에 맞는 도구를 우선 선택합니다.",
)

run = st.button("Agent 분석 실행", type="primary", use_container_width=True, disabled=not uploaded_files or not health)
if run:
    with st.spinner("Agent planner가 목표를 해석하고 필요한 CXR tool을 선택·실행 중입니다..."):
        result_payload = call_agent_api(uploaded_files, model_key, threshold, question)
        st.session_state["agent_result"] = result_payload
        st.session_state["agent_uploaded_lookup"] = {f.name: f.getvalue() for f in uploaded_files}
        if result_payload:
            chat_context = compact_chat_result_payload(result_payload)
            st.session_state["agent_chat_context"] = chat_context
            if question.strip():
                user_msg = {"role": "user", "content": question.strip()}
                initial_chat = call_agent_chat_api(question, chat_context, [user_msg])
                if initial_chat and initial_chat.get("answer"):
                    answer = str(initial_chat.get("answer"))
                    if initial_chat.get("fallback") and initial_chat.get("error"):
                        answer += f"\n\n[LLM fallback 사유] {initial_chat.get('error')}"
                    st.session_state["agent_chat_messages"] = [user_msg, {"role": "agent", "content": answer, "meta": initial_chat}]
                else:
                    st.session_state["agent_chat_messages"] = build_initial_chat(result_payload, question)
            else:
                st.session_state["agent_chat_messages"] = build_initial_chat(result_payload, question)
            st.session_state["agent_chat_nonce"] = st.session_state.get("agent_chat_nonce", 0) + 1

result = st.session_state.get("agent_result")
uploaded_lookup = st.session_state.get("agent_uploaded_lookup", {})

if not health:
    st.warning("백엔드가 연결되지 않았습니다. FastAPI를 실행한 뒤 다시 시도하세요.")
    st.code("uvicorn api.main:app --reload --port 8000", language="bash")
elif not result:
    st.markdown(
        """
<div class="agent-card">
    <h3>분석 대기 중</h3>
    <p>여러 장의 X-ray 또는 DICOM을 업로드한 뒤 Agent 분석을 실행하면, 이미지별 초안과 피드백 창구가 자동으로 생성됩니다.</p>
</div>
""",
        unsafe_allow_html=True,
    )
else:
    summary = result.get("agent_summary", {}) or {}
    cases = result.get("cases", [])

    st.markdown('<div class="section-title">Agent 전체 요약</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="agent-card">
    <h3>케이스 묶음 요약</h3>
    <p>{escape(str(summary.get('narrative', '')))}</p>
    <p class="small-muted">{escape(str(summary.get('safety_note', result.get('safety_note', ''))))}</p>
</div>
""",
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='metric-card'><div class='value'>{result.get('case_count', 0)}</div><div class='label'>분석 이미지</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-card'><div class='value'>{summary.get('placeholder_count', 0)}</div><div class='label'>Placeholder</div></div>", unsafe_allow_html=True)
    with m3:
        comparison_on = "ON" if (summary.get("comparison") or {}).get("enabled") else "OFF"
        st.markdown(f"<div class='metric-card'><div class='value'>{comparison_on}</div><div class='label'>비교 분석</div></div>", unsafe_allow_html=True)
    with m4:
        st.markdown(f"<div class='metric-card'><div class='value'>{result.get('model_key', '-')}</div><div class='label'>모델</div></div>", unsafe_allow_html=True)

    render_agent_execution_console(result)

    st.markdown('<div class="section-title">전체 영상 동시 분석</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="agent-card">
    <h3>업로드한 모든 영상의 분석 결과</h3>
    <p>여러 장을 업로드한 경우 첫 번째와 마지막 영상만 보지 않고, 모든 영상의 Top 소견·우선도·품질·Grad-CAM 여부·질환 확률을 한 번에 비교합니다.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    overview_df = build_all_case_overview_df(cases, float(result.get("threshold", threshold)))
    if not overview_df.empty:
        st.dataframe(overview_df, width="stretch", hide_index=True)

    with st.expander("질환별 확률 매트릭스 · 모든 영상 동시 보기", expanded=True):
        probability_df = build_all_case_probability_df(cases)
        if not probability_df.empty:
            st.dataframe(probability_df, width="stretch", hide_index=True)
        heatmap_fig = render_all_case_probability_heatmap(cases, float(result.get("threshold", threshold)))
        if heatmap_fig is not None:
            st.plotly_chart(heatmap_fig, width="stretch", config={"displayModeBar": False})

    comparison = summary.get("comparison") or {}
    if comparison.get("enabled"):
        st.markdown('<div class="section-title">첫 영상 ↔ 마지막 영상 변화량 참고</div>', unsafe_allow_html=True)
        st.markdown(f"<div class='agent-card'><p>{escape(str(comparison.get('summary', '')))}</p><p class='small-muted'>위의 전체 영상 동시 분석 테이블이 기본 비교 화면이며, 아래 표는 시간축이 있는 케이스에서 첫 영상과 마지막 영상의 변화량을 빠르게 확인하기 위한 보조 표입니다.</p></div>", unsafe_allow_html=True)
        deltas = comparison.get("probability_deltas", [])
        if deltas:
            delta_df = pd.DataFrame(deltas)
            delta_df = delta_df.rename(columns={
                "label_kr": "질환",
                "first_probability": "첫 영상 확률",
                "last_probability": "마지막 영상 확률",
                "delta": "변화량",
            })[["질환", "첫 영상 확률", "마지막 영상 확률", "변화량"]]
            st.dataframe(delta_df, width="stretch", hide_index=True)

    render_agent_chat(result)

    st.markdown('<div class="section-title">이미지별 판독·피드백</div>', unsafe_allow_html=True)
    if cases:
        tabs = st.tabs([f"{idx + 1}. {case.get('filename', 'case')}" for idx, case in enumerate(cases)])
        for tab, case in zip(tabs, cases):
            with tab:
                prediction = case.get("prediction", {}) or {}
                agent_profile = case.get("agent_profile", {}) or {}
                quality = agent_profile.get("quality_check", {}) or {}
                triage = agent_profile.get("triage_assessment", {}) or {}
                anatomy = agent_profile.get("anatomy_assessment", {}) or {}
                probs = case.get("probabilities", {}) or {}
                filename = case.get("filename", "uploaded")
                case_id = case.get("case_id", "CXR-UNKNOWN")
                triage_level = str(triage.get("triage_level", ""))
                triage_class = "demo" if case.get("is_placeholder") else ("urgent" if "URGENT" in triage_level else "high" if "HIGH" in triage_level else "normal")

                left, right = st.columns([1.15, 1.85], gap="large")
                with left:
                    preview = safe_preview_image(uploaded_lookup, filename)
                    st.markdown("#### 원본 이미지")
                    if preview is not None:
                        st.image(preview, width="stretch", caption=filename)
                    else:
                        st.info("DICOM 또는 미리보기 불가 파일입니다. Backend Agent가 DICOM 변환 후 분석했습니다.")
                    gradcam = prediction.get("GradCAM_Base64", "")
                    if gradcam and len(gradcam) > 500:
                        try:
                            st.markdown("#### Grad-CAM")
                            st.image(base64.b64decode(gradcam), width="stretch")
                        except Exception:
                            st.warning("Grad-CAM 이미지를 렌더링하지 못했습니다.")
                    else:
                        st.caption("실제 모델 Grad-CAM이 없거나 Placeholder 응답입니다.")

                with right:
                    st.markdown(
                        f"""
<div class="triage {triage_class}">
    <h4>{escape(str(triage.get('triage_label_kr', 'Agent 판정')))} · {escape(str(case.get('top_disease', '-')).replace('_', ' '))} {float(case.get('top_probability', 0.0)):.1%}</h4>
    <p>{escape(str(triage.get('reason', '')))}</p>
</div>
""",
                        unsafe_allow_html=True,
                    )
                    tag_html = " ".join(
                        f"<span class='disease-tag'>{escape(DISEASE_LABELS_KR.get(d, d))} {float(probs.get(d, 0.0)):.0%}</span>"
                        for d in case.get("detected_diseases", [])
                    ) or "<span class='disease-tag'>임계값 이상 소견 없음</span>"
                    st.markdown(tag_html, unsafe_allow_html=True)
                    st.plotly_chart(render_bar_chart(probs, float(result.get("threshold", threshold))), width="stretch", config={"displayModeBar": False})

                st.markdown("#### 이미지별 AI 판독문 초안")
                edited_report = st.text_area(
                    "판독문 초안 편집",
                    value=case.get("report_draft", ""),
                    height=210,
                    key=f"agent_report_{case_id}",
                )
                st.download_button(
                    "이 이미지의 판독문 초안 다운로드",
                    data=edited_report.encode("utf-8"),
                    file_name=f"{case_id}_agent_report.txt",
                    mime="text/plain",
                    key=f"agent_download_{case_id}",
                )

                info_cols = st.columns(3)
                with info_cols[0]:
                    st.markdown("##### 품질 점검")
                    st.metric("품질 등급", quality.get("quality_grade", "-"), f"{quality.get('quality_score', '-')}/100")
                    for flag in quality.get("flags", [])[:3]:
                        st.caption(f"• {flag}")
                with info_cols[1]:
                    st.markdown("##### 해부학 ROI")
                    rois = anatomy.get("focus_rois", [])
                    if rois:
                        for roi in rois[:4]:
                            st.caption(f"• {roi.get('label_kr')} · {float(roi.get('priority_score', 0.0)):.0%}")
                    else:
                        st.caption("ROI 스캐폴드 없음")
                with info_cols[2]:
                    st.markdown("##### DICOM/메타데이터")
                    meta = prediction.get("Image_Metadata", {}) or {}
                    st.caption(f"크기: {meta.get('width', '-') } × {meta.get('height', '-')}")
                    st.caption(f"DICOM 입력: {'예' if meta.get('is_dicom_input') else '아니오'}")
                    if meta.get("dicom_metadata"):
                        with st.expander("DICOM 메타데이터 보기"):
                            st.json(meta.get("dicom_metadata"))

                with st.expander("해부학 ROI 상세 보기"):
                    st.caption(anatomy.get("disclaimer", ""))
                    roi_rows = []
                    for roi in anatomy.get("focus_rois", []):
                        roi_rows.append({
                            "ROI": roi.get("label_kr"),
                            "관련 소견": ", ".join(DISEASE_LABELS_KR.get(d, d) for d in roi.get("related_findings", [])),
                            "우선도": roi.get("priority_score"),
                            "검토 힌트": roi.get("review_hint"),
                            "bbox_px": roi.get("bbox_px"),
                        })
                    if roi_rows:
                        st.dataframe(pd.DataFrame(roi_rows), width="stretch", hide_index=True)

                st.markdown("#### 이 이미지에 대한 의료진 피드백")
                fb_left, fb_right = st.columns([1, 1])
                with fb_left:
                    reviewer_id = st.text_input("판독의/검수자 ID", key=f"agent_reviewer_{case_id}", placeholder="예: RAD01")
                    corrected_labels = st.multiselect(
                        "수정 라벨",
                        DISEASE_LABELS,
                        format_func=lambda x: f"{DISEASE_LABELS_KR.get(x, x)} / {x.replace('_', ' ')}",
                        key=f"agent_corrected_{case_id}",
                    )
                with fb_right:
                    comment = st.text_area("판독의 코멘트", key=f"agent_comment_{case_id}", height=105)

                fb_cols = st.columns(5)
                for fb_col, feedback_type in zip(fb_cols, FEEDBACK_TYPES):
                    with fb_col:
                        if st.button(feedback_type, key=f"agent_fb_{feedback_type}_{case_id}", use_container_width=True):
                            if feedback_type == "질환 라벨 수정" and not corrected_labels:
                                st.warning("라벨 수정 피드백에는 수정 라벨이 필요합니다.")
                            elif feedback_type == "판독의 코멘트" and not comment.strip():
                                st.warning("코멘트를 입력해야 저장할 수 있습니다.")
                            else:
                                submit_feedback(case, feedback_type, float(result.get("threshold", threshold)), corrected_labels, comment, reviewer_id, edited_report)
