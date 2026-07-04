"""
CXR-CAD — Professional Medical Imaging Dashboard.

Streamlit 기반 프론트엔드.
FastAPI 백엔드와 HTTP 통신.
DenseNet-121 / EfficientNet-B4 / ViT-B/16 모델 선택 UI 포함.
"""

from __future__ import annotations

import io
import hashlib
import os
from html import escape

import requests
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# ── 설정 ─────────────────────────────────────────────────────────────────────
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

FEEDBACK_TYPES = [
    "AI 판단 동의",
    "AI 판단 불일치",
    "히트맵 위치 부정확",
    "질환 라벨 수정",
    "판독의 코멘트",
]

MODEL_OPTIONS = {
    "ensemble":     {"label": "Ensemble (Recommended)", "params": "Combined", "tag": "최고 성능"},
    "densenet":     {"label": "DenseNet-121", "params": "~8M", "tag": "가볍고 빠름"},
    "efficientnet": {"label": "EfficientNet-B4", "params": "~19M", "tag": "균형 최적화"},
    "vit":          {"label": "ViT-B/16", "params": "~86M", "tag": "전역 문맥 학습"},
}

# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CXR-CAD | Chest X-ray AI Diagnosis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --navy: #0f172a;
        --slate: #1e293b;
        --blue: #2563eb;
        --cyan: #06b6d4;
        --teal: #14b8a6;
        --violet: #7c3aed;
        --rose: #e11d48;
        --amber: #f59e0b;
        --card: rgba(255,255,255,0.92);
        --line: rgba(148,163,184,0.25);
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 12% 8%, rgba(14,165,233,0.16), transparent 28%),
            radial-gradient(circle at 88% 18%, rgba(124,58,237,0.13), transparent 28%),
            linear-gradient(180deg, #f8fbff 0%, #edf5ff 50%, #f8fafc 100%);
    }
    .main .block-container { padding-top: 1.4rem; padding-bottom: 1rem; max-width: 1420px; }

    [data-testid="stSidebar"] {
        background:
            radial-gradient(circle at top left, rgba(56,189,248,0.22), transparent 30%),
            linear-gradient(180deg, #08111f 0%, #0f172a 48%, #111827 100%);
        border-right: 1px solid rgba(148,163,184,0.22);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        color: #e5edf8 !important;
    }
    [data-testid="stSidebarNav"] span {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] hr { border-color: rgba(148,163,184,0.22) !important; }
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSlider label { color: #d7e3f4 !important; }

    /* Sidebar form controls: dark text on light input boxes */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        background: #f8fafc !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        border-color: rgba(148,163,184,0.45) !important;
        opacity: 1 !important;
        font-weight: 650 !important;
    }
    [data-testid="stSidebar"] input:disabled,
    [data-testid="stSidebar"] textarea:disabled {
        color: #334155 !important;
        -webkit-text-fill-color: #334155 !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] input::placeholder,
    [data-testid="stSidebar"] textarea::placeholder {
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #f8fafc !important;
        border-color: rgba(148,163,184,0.45) !important;
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] div {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] svg {
        color: #475569 !important;
        fill: #475569 !important;
    }
    div[data-baseweb="popover"] div[role="listbox"] { background: #ffffff !important; }
    div[data-baseweb="popover"] div[role="option"] { color: #0f172a !important; background: #ffffff !important; }
    div[data-baseweb="popover"] div[role="option"]:hover { background: #e0f2fe !important; color: #0f172a !important; }

    .brand-block {
        border: 1px solid rgba(125,211,252,0.22);
        border-radius: 18px;
        padding: 1rem;
        background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,58,138,0.5));
        box-shadow: 0 18px 48px rgba(8,17,31,0.35);
    }
    .brand-mark {
        width: 48px; height: 48px; border-radius: 16px;
        display: flex; align-items: center; justify-content: center;
        background: linear-gradient(135deg, #0ea5e9, #14b8a6);
        color: white !important; font-weight: 800; letter-spacing: -0.04em;
        margin-bottom: 0.75rem;
        box-shadow: 0 12px 28px rgba(14,165,233,0.28);
    }
    .brand-title { font-size: 1.15rem; font-weight: 800; margin: 0; color: white !important; }
    .brand-subtitle { font-size: 0.78rem; line-height: 1.45; color: #b6c7dc !important; margin-top: 0.25rem; }

    .api-status-card {
        border-radius: 18px;
        padding: 1rem;
        border: 1px solid rgba(148,163,184,0.22);
        box-shadow: 0 16px 36px rgba(8,17,31,0.24);
        margin: 0.8rem 0 0.35rem;
    }
    .api-status-card.live { background: linear-gradient(135deg, rgba(6,95,70,0.96), rgba(15,118,110,0.72)); border-color: rgba(94,234,212,0.34); }
    .api-status-card.demo { background: linear-gradient(135deg, rgba(30,58,138,0.96), rgba(67,56,202,0.70)); border-color: rgba(147,197,253,0.34); }
    .api-status-card.offline { background: linear-gradient(135deg, rgba(127,29,29,0.96), rgba(154,52,18,0.70)); border-color: rgba(254,202,202,0.34); }
    .status-kicker { font-size: 0.68rem; letter-spacing: 0.12em; text-transform: uppercase; opacity: 0.75; font-weight: 700; }
    .status-head { font-size: 1.05rem; font-weight: 800; margin-top: 0.3rem; color: white !important; display: flex; align-items: center; gap: 0.5rem; }
    .status-dot { width: 0.65rem; height: 0.65rem; border-radius: 999px; background: #67e8f9; display: inline-block; box-shadow: 0 0 0 5px rgba(103,232,249,0.16); }
    .offline .status-dot { background: #fecaca; box-shadow: 0 0 0 5px rgba(254,202,202,0.14); }
    .status-copy { margin-top: 0.45rem; font-size: 0.78rem; line-height: 1.45; color: #dbeafe !important; }
    .status-grid { display: grid; grid-template-columns: 1fr; gap: 0.35rem; margin-top: 0.75rem; }
    .status-row { display: flex; justify-content: space-between; gap: 0.5rem; font-size: 0.72rem; padding: 0.42rem 0.5rem; border-radius: 10px; background: rgba(255,255,255,0.08); }
    .status-row span { color: #cbd5e1 !important; }
    .status-row b { color: #ffffff !important; font-weight: 700; text-align: right; }

    .model-card {
        background: linear-gradient(135deg, rgba(30,41,59,0.95) 0%, rgba(15,23,42,0.96) 100%);
        border: 1px solid rgba(148,163,184,0.22);
        border-radius: 16px;
        padding: 1rem;
        margin: 0.45rem 0;
        transition: border-color 0.2s, transform 0.2s;
        box-shadow: 0 12px 30px rgba(8,17,31,0.20);
    }
    .model-card.selected {
        border-color: rgba(56,189,248,0.65);
        box-shadow: 0 0 0 2px rgba(56,189,248,0.17), 0 18px 42px rgba(8,17,31,0.28);
    }
    .model-card .model-name { font-weight: 800; font-size: 0.96rem; color: #f8fafc !important; margin: 0; }
    .model-card .model-meta { font-size: 0.77rem; color: #b6c7dc !important; margin: 0.35rem 0 0 0; line-height: 1.4; }
    .model-tag {
        display: inline-block;
        background: linear-gradient(135deg, rgba(14,165,233,0.22), rgba(20,184,166,0.20));
        color: #bae6fd !important;
        border: 1px solid rgba(125,211,252,0.32);
        border-radius: 999px;
        padding: 0.14rem 0.55rem;
        font-size: 0.68rem;
        margin-left: 0.4rem;
        vertical-align: middle;
    }

    .premium-card, .glass-card {
        background: var(--card);
        border: 1px solid rgba(148,163,184,0.24);
        border-radius: 22px;
        padding: 1.45rem;
        box-shadow: 0 18px 42px rgba(15,23,42,0.08);
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
    }
    .glass-card.blue { background: linear-gradient(135deg, rgba(239,246,255,0.95), rgba(224,242,254,0.88)); }
    .glass-card.teal { background: linear-gradient(135deg, rgba(240,253,250,0.95), rgba(204,251,241,0.80)); }
    .glass-card.violet { background: linear-gradient(135deg, rgba(245,243,255,0.96), rgba(237,233,254,0.84)); }

    .main-header {
        background:
            radial-gradient(circle at 85% 18%, rgba(125,211,252,0.25), transparent 24%),
            linear-gradient(135deg, #08111f 0%, #15345f 52%, #0f766e 100%);
        border: 1px solid rgba(125,211,252,0.24);
        border-radius: 28px;
        padding: 1.65rem 2rem;
        margin-bottom: 1.4rem;
        color: white;
        box-shadow: 0 24px 58px rgba(15,23,42,0.20);
    }
    .main-header .eyebrow { color:#a7f3d0 !important; font-size:0.72rem; letter-spacing:0.16em; text-transform:uppercase; font-weight:800; margin-bottom:0.35rem; }
    .main-header h1 { margin:0; font-size:1.75rem; font-weight:800; color:white !important; letter-spacing:-0.03em; }
    .main-header p  { margin:0.35rem 0 0; font-size:0.9rem; line-height:1.45; color:#cbd5e1 !important; max-width: 900px; }
    .hero-meta { display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.9rem; }
    .hero-pill { border:1px solid rgba(226,232,240,0.22); background:rgba(255,255,255,0.08); color:#e2e8f0 !important; border-radius:999px; padding:0.38rem 0.72rem; font-size:0.76rem; font-weight:700; }

    .section-title {
        font-size: 1.02rem; font-weight: 800; color: #0f172a;
        margin-bottom: 0.75rem; padding-bottom: 0.6rem;
        border-bottom: 1px solid rgba(148,163,184,0.28);
        display:flex; align-items:center; gap:0.5rem;
    }
    .section-title::before {
        content:""; width:0.55rem; height:0.55rem; border-radius:999px;
        background: linear-gradient(135deg, var(--blue), var(--teal));
        box-shadow: 0 0 0 5px rgba(14,165,233,0.12);
    }

    .upload-hero {
        background:
            radial-gradient(circle at 18% 20%, rgba(14,165,233,0.18), transparent 28%),
            radial-gradient(circle at 82% 18%, rgba(124,58,237,0.14), transparent 24%),
            linear-gradient(135deg, rgba(255,255,255,0.96), rgba(239,246,255,0.92));
        border: 1px solid rgba(56,189,248,0.28);
        border-radius: 28px;
        padding: 2rem;
        box-shadow: 0 22px 56px rgba(15,23,42,0.10);
        margin-bottom: 1rem;
    }
    .upload-kicker { color:#0369a1 !important; font-size:0.75rem; font-weight:800; letter-spacing:0.14em; text-transform:uppercase; }
    .upload-title { color:#0f172a !important; font-size:1.55rem; font-weight:850; margin:0.35rem 0 0; letter-spacing:-0.03em; }
    .upload-desc { color:#475569 !important; font-size:0.94rem; line-height:1.55; max-width:760px; margin-top:0.55rem; }
    .upload-steps { display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:0.8rem; margin-top:1.15rem; }
    .upload-step { border:1px solid rgba(148,163,184,0.22); background:rgba(255,255,255,0.75); border-radius:18px; padding:1rem; }
    .upload-step b { color:#0f172a !important; font-size:0.9rem; }
    .upload-step p { color:#64748b !important; font-size:0.78rem; margin:0.28rem 0 0; line-height:1.45; }

    [data-testid="stFileUploader"] { width: 100%; }
    [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, rgba(240,249,255,0.98), rgba(236,253,245,0.92)) !important;
        border: 2px dashed rgba(14,165,233,0.70) !important;
        border-radius: 20px !important;
        min-height: 118px !important;
        padding: 1.2rem !important;
        transition: all 0.2s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(20,184,166,0.95) !important;
        box-shadow: 0 16px 32px rgba(14,165,233,0.12);
        transform: translateY(-1px);
    }
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #2563eb, #0891b2) !important;
        color: white !important;
        border-radius: 999px !important;
        border: 0 !important;
        padding: 0.55rem 1.1rem !important;
        font-weight: 800 !important;
        box-shadow: 0 10px 22px rgba(37,99,235,0.22);
    }

    .top-disease-card {
        background:
            radial-gradient(circle at 12% 12%, rgba(254,202,202,0.28), transparent 22%),
            linear-gradient(135deg, #991b1b 0%, #e11d48 52%, #7f1d1d 100%);
        border-radius: 24px;
        padding: 1.55rem;
        color: white;
        text-align: center;
        box-shadow: 0 18px 42px rgba(225,29,72,0.28);
        margin-bottom: 1rem;
        border: 1px solid rgba(254,202,202,0.32);
    }
    .top-disease-card h2 { margin: 0; font-size: 1.8rem; font-weight: 800; color: white !important; }
    .top-disease-card .prob { font-size: 3rem; font-weight: 300; margin: 0.25rem 0; color: #fff1f2 !important; }
    .top-disease-card .sub  { font-size: 0.72rem; letter-spacing: 0.18em; opacity: 0.82; color: #ffe4e6 !important; font-weight:800; }

    .disease-tag { display:inline-block; background:linear-gradient(135deg,#fff1f2,#ffe4e6); color:#9f1239; padding:0.45rem 0.95rem; border-radius:999px; font-size:0.84rem; font-weight:750; margin:0.22rem; border:1px solid #fecdd3; }
    .inference-metric { background:linear-gradient(135deg,#eff6ff,#ecfeff); border:1px solid rgba(125,211,252,0.55); border-radius:18px; padding:1rem; text-align:center; box-shadow:0 12px 26px rgba(14,165,233,0.08); }
    .inference-metric .value { font-size:1.5rem; font-weight:850; color:#0f4c81; }
    .inference-metric .label { font-size:0.72rem; text-transform:uppercase; letter-spacing:1.5px; color:#0369a1; font-weight:800; }

    .mode-card {
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(148,163,184,0.24);
        box-shadow: 0 14px 30px rgba(15,23,42,0.08);
    }
    .mode-card.live { background: linear-gradient(135deg, #ecfdf5, #dbeafe); border-color: rgba(20,184,166,0.32); }
    .mode-card.demo { background: linear-gradient(135deg, #fff7ed, #fef3c7); border-color: rgba(245,158,11,0.38); }
    .mode-card h4 { margin: 0; color: #0f172a !important; font-size: 1rem; font-weight: 850; }
    .mode-card p { margin: 0.32rem 0 0; color: #475569 !important; font-size: 0.86rem; line-height: 1.48; }

    .feature-card { min-height: 170px; }
    .feature-tag { display:inline-block; border-radius:999px; padding:0.28rem 0.58rem; font-size:0.72rem; font-weight:800; margin-bottom:0.7rem; color:#075985 !important; background:#e0f2fe; }
    .feature-card h4 { color:#0f172a !important; margin:0 0 0.45rem; font-size:1rem; }
    .feature-card p { color:#64748b !important; font-size:0.84rem; margin:0; line-height:1.5; }

    .clinical-report-card {
        border-radius: 22px;
        padding: 1.15rem;
        margin: 1rem 0;
        background:
            radial-gradient(circle at 12% 0%, rgba(20,184,166,0.16), transparent 26%),
            linear-gradient(135deg, rgba(255,255,255,0.98), rgba(240,253,250,0.92));
        border: 1px solid rgba(20,184,166,0.28);
        box-shadow: 0 16px 34px rgba(15,23,42,0.08);
    }
    .clinical-report-card h4 { margin:0; color:#0f172a !important; font-weight:850; font-size:1rem; }
    .clinical-report-card p { margin:0.4rem 0 0; color:#475569 !important; font-size:0.84rem; line-height:1.55; }
    .review-note {
        border-radius: 16px;
        padding: 0.9rem 1rem;
        background: linear-gradient(135deg, #fff7ed, #fefce8);
        border: 1px solid rgba(245,158,11,0.30);
        color: #78350f !important;
        font-size: 0.84rem;
        line-height: 1.5;
        font-weight: 700;
    }
    .feedback-card {
        border-radius: 22px;
        padding: 1.15rem;
        margin: 1rem 0;
        background:
            radial-gradient(circle at 8% 18%, rgba(37,99,235,0.14), transparent 24%),
            linear-gradient(135deg, rgba(255,255,255,0.98), rgba(239,246,255,0.94));
        border: 1px solid rgba(59,130,246,0.28);
        box-shadow: 0 16px 34px rgba(15,23,42,0.08);
    }
    .feedback-card h4 { margin:0; color:#0f172a !important; font-weight:850; font-size:1rem; }
    .feedback-card p { margin:0.4rem 0 0; color:#475569 !important; font-size:0.84rem; line-height:1.55; }
    .queue-badge { display:inline-block; border-radius:999px; padding:0.28rem 0.62rem; background:#dbeafe; color:#1e3a8a !important; font-size:0.74rem; font-weight:850; margin-top:0.65rem; }

    #MainMenu {visibility:hidden;} footer {visibility:hidden;}
    @media (max-width: 900px) { .upload-steps { grid-template-columns: 1fr; } }
</style>
""", unsafe_allow_html=True)


# ── 헬퍼 함수 ─────────────────────────────────────────────────────────────────

def check_api_health() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def get_model_info_from_api() -> dict:
    try:
        r = requests.get(f"{API_URL}/models", timeout=3)
        if r.status_code == 200:
            return r.json().get("models", {})
    except Exception:
        pass
    return {}


def call_predict_api(image_bytes: bytes, filename: str, model_key: str, threshold: float) -> dict | None:
    ext = filename.lower().split(".")[-1]
    content_type = "image/png" if ext == "png" else "image/jpeg"
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            params={"model": model_key, "threshold": threshold},
            files={"file": (filename, image_bytes, content_type)},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()
        st.error(f"API 오류 {resp.status_code}: {resp.text}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("백엔드 API에 연결할 수 없습니다.")
        return None


def call_feedback_api(payload: dict) -> dict | None:
    try:
        resp = requests.post(f"{API_URL}/feedback", json=payload, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        st.error(f"피드백 저장 오류 {resp.status_code}: {resp.text}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("백엔드 API에 연결할 수 없어 피드백을 저장하지 못했습니다.")
        return None


def get_feedback_queue_summary(limit: int = 5) -> dict | None:
    try:
        resp = requests.get(f"{API_URL}/feedback/queue", params={"limit": limit}, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def disease_display(label: str) -> str:
    return f"{DISEASE_LABELS_KR.get(label, label.replace('_', ' '))} / {label.replace('_', ' ')}"


def build_prediction_summary(result: dict, probs: dict, filename: str, edited_report: str) -> dict:
    return {
        "filename": filename,
        "top_disease": result.get("Top_Disease"),
        "top_probability": result.get("Top_Probability"),
        "detected_diseases": result.get("Detected_Diseases", []),
        "probabilities": probs,
        "model_used": result.get("Model_Used"),
        "model_key": result.get("Model_Key"),
        "is_placeholder": result.get("Is_Placeholder"),
        "report_draft_kr": edited_report,
        "need_review_reason": result.get("Need_Review_Reason", ""),
    }


def submit_clinician_feedback(
    *,
    feedback_type: str,
    result: dict,
    probs: dict,
    filename: str,
    threshold: float,
    corrected_labels: list[str],
    comment: str,
    reviewer_id: str,
    edited_report: str,
) -> None:
    payload = {
        "case_id": result.get("Case_ID", "CXR-UNKNOWN"),
        "feedback_type": feedback_type,
        "original_top_disease": result.get("Top_Disease"),
        "corrected_labels": corrected_labels,
        "comment": comment.strip(),
        "reviewer_id": reviewer_id.strip() or None,
        "model_key": result.get("Model_Key"),
        "threshold": threshold,
        "prediction_summary": build_prediction_summary(result, probs, filename, edited_report),
    }
    saved = call_feedback_api(payload)
    if saved:
        st.session_state["last_feedback_response"] = saved
        st.success(f"{saved.get('message', '피드백이 저장되었습니다')} 큐 ID: {saved.get('queue_id', '-')}")


def get_risk_color(prob: float, threshold: float) -> str:
    if prob >= 0.5:
        return "#e11d48"
    elif prob >= threshold:
        return "#f59e0b"
    else:
        return "#14b8a6"


def create_disease_chart(probs: dict, threshold: float) -> go.Figure:
    sorted_items = sorted(probs.items(), key=lambda x: x[1])
    diseases = [k.replace("_", " ") for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = [get_risk_color(v, threshold) for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=diseases,
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
        annotation=dict(text=f"임계값 ({threshold:.0%})", font=dict(size=10, color="#b45309"), yref="paper", y=1.05),
    )
    fig.update_layout(
        height=480,
        margin=dict(l=0, r=45, t=30, b=20),
        plot_bgcolor="rgba(255,255,255,0)",
        paper_bgcolor="rgba(255,255,255,0)",
        xaxis=dict(
            range=[0, 1.12],
            showgrid=True,
            gridcolor="rgba(148,163,184,0.18)",
            tickformat=".0%",
            tickfont=dict(size=10, family="Inter", color="#64748b"),
        ),
        yaxis=dict(tickfont=dict(size=12, family="Inter", color="#334155")),
        font=dict(family="Inter"),
    )
    return fig


def render_status_card(health: dict | None) -> None:
    if health:
        loaded = health.get("loaded_models", [])
        model_ver = health.get("model_version", "")
        if loaded:
            state_class = "live"
            state_title = "FastAPI 연결 · 실제 모델 추론"
            state_copy = "업로드 이미지는 /predict 엔드포인트에서 로드된 체크포인트 가중치로 추론됩니다."
            model_count = f"{len(loaded)}개 로드"
            mode = "Real inference"
        else:
            state_class = "demo"
            state_title = "FastAPI 연결 · Placeholder 모드"
            state_copy = "서버는 연결되어 있지만 체크포인트가 없어 데모 확률을 반환합니다. 실제 추론처럼 오해되지 않도록 결과 화면에도 표시됩니다."
            model_count = "0개 로드"
            mode = "Demo response"
    else:
        state_class = "offline"
        state_title = "FastAPI 미연결"
        state_copy = "백엔드 서버를 실행해야 이미지 분석을 시작할 수 있습니다."
        model_ver = "연결 안 됨"
        model_count = "확인 불가"
        mode = "Offline"

    st.markdown(
        f"""
        <div class="api-status-card {state_class}">
            <div class="status-kicker">API STATUS</div>
            <div class="status-head"><span class="status-dot"></span>{state_title}</div>
            <div class="status-copy">{state_copy}</div>
            <div class="status-grid">
                <div class="status-row"><span>Endpoint</span><b>{API_URL}/predict</b></div>
                <div class="status-row"><span>Model weights</span><b>{model_count}</b></div>
                <div class="status-row"><span>Result mode</span><b>{mode}</b></div>
                <div class="status-row"><span>Version</span><b>{model_ver or '-'}</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


ACTIVE_UPLOAD_KEY = "active_xray_upload"


def persist_uploaded_image(uploaded_file) -> dict | None:
    """Keep the uploaded image available after Streamlit page navigation.

    Streamlit's file_uploader value can be cleared when a multipage app switches
    away from the page that owns the widget.  Storing the immutable bytes in
    session_state prevents the uploaded X-ray and its cached prediction from
    disappearing when the user opens Reliability Readiness or Result Analysis
    and then returns to the main app page.
    """
    if uploaded_file is None:
        return st.session_state.get(ACTIVE_UPLOAD_KEY)

    image_bytes = uploaded_file.getvalue()
    if not image_bytes:
        return st.session_state.get(ACTIVE_UPLOAD_KEY)

    image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
    active_upload = {
        "name": uploaded_file.name,
        "mime_type": getattr(uploaded_file, "type", "image/*") or "image/*",
        "bytes": image_bytes,
        "hash": image_hash,
    }
    previous = st.session_state.get(ACTIVE_UPLOAD_KEY)
    st.session_state[ACTIVE_UPLOAD_KEY] = active_upload

    # A different image invalidates only image-dependent widgets and results.
    if not previous or previous.get("hash") != image_hash or previous.get("name") != uploaded_file.name:
        st.session_state.pop("result_cache_key", None)
        st.session_state.pop("last_prediction_result", None)

    return active_upload


def clear_active_upload() -> None:
    for key in [ACTIVE_UPLOAD_KEY, "result_cache_key", "last_prediction_result", "xray_uploader_main"]:
        st.session_state.pop(key, None)



# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="brand-block">
            <div class="brand-mark">CXR</div>
            <p class="brand-title">CXR-CAD</p>
            <div class="brand-subtitle">Chest X-ray AI diagnosis dashboard<br>FastAPI inference workflow</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    health = check_api_health()
    api_model_info = get_model_info_from_api() if health else {}
    render_status_card(health)

    st.divider()
    st.markdown("### 모델 선택")

    model_labels = [v["label"] for v in MODEL_OPTIONS.values()]
    model_keys = list(MODEL_OPTIONS.keys())

    selected_label = st.radio(
        "분석에 사용할 모델",
        options=model_labels,
        label_visibility="collapsed",
        key="model_radio",
    )
    selected_model_key = model_keys[model_labels.index(selected_label)]
    model_opt = MODEL_OPTIONS[selected_model_key]

    is_loaded = api_model_info.get(selected_model_key, {}).get("is_loaded", False)
    loaded_badge = "실제 가중치 로드" if is_loaded else "Placeholder 응답"
    api_desc = api_model_info.get(selected_model_key, {}).get("description", "")
    st.markdown(
        f"""
        <div class="model-card selected">
            <p class="model-name">{model_opt['label']}
                <span class="model-tag">{model_opt['tag']}</span>
            </p>
            <p class="model-meta">파라미터: {model_opt['params']} &nbsp;|&nbsp; {loaded_badge}</p>
            {"<p class='model-meta'>" + api_desc + "</p>" if api_desc else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### 판정 설정")
    threshold = st.slider(
        "감지 임계값",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05,
        help="이 확률 이상의 질환을 '감지됨'으로 분류합니다.",
        key="threshold_slider",
    )

    st.divider()
    st.page_link("pages/agent_workbench.py", label="Agentic Case Workbench")
    st.page_link("pages/analysis_results.py", label="상세 분석 결과 보기")

    st.divider()
    st.markdown(
        "<div style='text-align:center;opacity:0.55;font-size:0.72rem;'>"
        "CXR-CAD v0.2.0<br>For Research Use Only</div>",
        unsafe_allow_html=True,
    )

active_upload = st.session_state.get(ACTIVE_UPLOAD_KEY)

# ── 메인 헤더 ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="eyebrow">Clinical AI Workflow</div>
    <h1>CXR-CAD — Chest X-ray AI Diagnosis</h1>
    <p>흉부 X-ray 이미지를 업로드하면 FastAPI 백엔드가 선택한 모델로 14개 흉부 질환 확률을 계산하고, 실제 모델 추론인지 Placeholder 데모 응답인지 명확히 표시합니다.</p>
    <div class="hero-meta">
        <span class="hero-pill">DenseNet-121</span>
        <span class="hero-pill">EfficientNet-B4</span>
        <span class="hero-pill">ViT-B/16</span>
        <span class="hero-pill">FastAPI /predict</span>
        <span class="hero-pill">Agent /agent/analyze</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── 이미지 업로드 안내 및 메인 업로더 ─────────────────────────────────────────
if active_upload is None:
    st.markdown("""
    <div class="upload-hero">
        <div class="upload-kicker">Image input</div>
        <div class="upload-title">아래 업로드 박스를 클릭하거나 이미지를 끌어다 놓으세요.</div>
        <div class="upload-desc">
            선택한 파일은 FastAPI의 <b>/predict</b> 엔드포인트로 전달되며, 체크포인트가 로드된 경우 실제 모델 추론 결과를, 체크포인트가 없으면 Placeholder 데모 응답을 반환합니다.
        </div>
        <div class="upload-steps">
            <div class="upload-step"><b>1. 모델 선택</b><p>왼쪽 사이드바에서 Ensemble, DenseNet, EfficientNet, ViT 중 하나를 선택합니다.</p></div>
            <div class="upload-step"><b>2. 이미지 업로드</b><p>PNG 또는 JPEG 흉부 X-ray 이미지를 선택합니다.</p></div>
            <div class="upload-step"><b>3. 추론 모드 확인</b><p>결과 카드에서 실제 추론인지 Placeholder인지 확인합니다.</p></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    upload_left, upload_center, upload_right = st.columns([1, 2, 1])
    with upload_center:
        main_uploaded_file = st.file_uploader(
            "X-ray 이미지 선택 또는 이 영역에 드래그 앤 드롭",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
            help="PA/AP 전면 흉부 X-ray 이미지(PNG 또는 JPEG)",
            key="xray_uploader_main",
        )
    active_upload = persist_uploaded_image(main_uploaded_file)
else:
    st.markdown(
        f"""
        <div class="glass-card blue" style="margin-bottom:1rem;">
            <div class="section-title">현재 보존된 분석 이미지</div>
            <p style="color:#475569;margin:0;line-height:1.6;">
                <b>{escape(active_upload.get('name', 'uploaded image'))}</b> · 화면을 Result Analysis 또는 Reliability Readiness로 전환해도
                업로드 이미지와 마지막 분석 결과가 세션에 유지됩니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("다른 이미지 업로드하기", key="clear_active_upload", use_container_width=False):
        clear_active_upload()
        st.rerun()

# ── 메인 콘텐츠 ───────────────────────────────────────────────────────────────
if active_upload is None:
    feat_cols = st.columns(3)
    features = [
        ("Architecture", "DenseNet-121", "Dense connectivity 기반 경량 모델. 빠른 추론과 안정적 성능."),
        ("Agentic", "Multi-image Workbench", "MedRAX식 다중 이미지·DICOM·비교 요약·이미지별 피드백 워크플로우."),
        ("Context", "ViT-B/16", "Self-Attention으로 영상 전역의 관계를 학습하는 모델."),
    ]
    for col, (tag, title, desc) in zip(feat_cols, features):
        with col:
            st.markdown(
                f"""
                <div class="premium-card feature-card">
                    <span class="feature-tag">{tag}</span>
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    health_text = "실제 모델 추론 가능" if health and health.get("loaded_models") else "Placeholder 또는 미연결"
    api_text = "FastAPI가 연결되어 있습니다." if health else "FastAPI 서버가 아직 연결되지 않았습니다."
    st.markdown(
        f"""
        <div class="glass-card blue">
            <div class="section-title">현재 실행 상태</div>
            <p style="color:#475569;margin:0;line-height:1.6;">
                {api_text} 현재 표시 모드는 <b>{health_text}</b>입니다. 
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    # ── 이미지 업로드 완료 → 분석 ─────────────────────────────────────────────
    image_bytes = active_upload["bytes"]
    filename = active_upload["name"]
    image_hash = active_upload["hash"]
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result_cache_key = f"{filename}:{image_hash}:{selected_model_key}:{threshold:.2f}"

    result = None
    if health:
        if st.session_state.get("result_cache_key") == result_cache_key:
            result = st.session_state.get("last_prediction_result")
        else:
            model_label = MODEL_OPTIONS[selected_model_key]["label"]
            with st.spinner(f"{model_label} 모델로 분석 중..."):
                result = call_predict_api(image_bytes, filename, selected_model_key, threshold)
            st.session_state["result_cache_key"] = result_cache_key
            st.session_state["last_prediction_result"] = result

    col_left, col_right = st.columns([2, 3], gap="large")

    # ── 좌측: 이미지 ─────────────────────────────────────────────────────────
    with col_left:
        st.markdown('<div class="section-title">업로드된 이미지</div>', unsafe_allow_html=True)
        st.markdown('<div class="premium-card" style="padding:1.2rem; text-align:center;">', unsafe_allow_html=True)
        st.image(image, width="stretch", caption=filename)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Grad-CAM 시각화</div>', unsafe_allow_html=True)

        is_placeholder_cam = True
        if result and "GradCAM_Base64" in result and len(result["GradCAM_Base64"]) > 500:
            is_placeholder_cam = False

        if not is_placeholder_cam:
            import base64
            cam_bytes = base64.b64decode(result["GradCAM_Base64"])
            st.markdown('<div class="premium-card" style="padding:1.2rem; text-align:center;">', unsafe_allow_html=True)
            st.image(cam_bytes, width="stretch")
            st.markdown("""
            <div class="glass-card blue" style="margin-top:1rem;margin-bottom:0;">
                <b>활성화 맵 해석 방법</b><br>
                <span style="color:#e11d48;font-weight:800;">붉은색</span> 영역은 AI가 해당 질환을 판단할 때 상대적으로 강하게 주목한 부위를 나타냅니다.
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="premium-card" style="text-align:center;padding:2.4rem 1.6rem;">
                <p style="color:#0f172a;font-weight:800;margin:0;">히트맵이 아직 준비되지 않았습니다</p>
                <p style="color:#64748b;font-size:0.86rem;margin:0.5rem 0 0;line-height:1.5;">
                    실제 학습 가중치가 적용된 모델 모드에서만 활성화 맵 렌더링이 지원됩니다.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ── 우측: 분석 결과 ──────────────────────────────────────────────────────
    with col_right:
        st.markdown('<div class="section-title">분석 결과</div>', unsafe_allow_html=True)

        if not health:
            st.error("백엔드 API가 연결되지 않았습니다. 서버를 먼저 실행하세요.")
            st.code("uvicorn api.main:app --reload --port 8000", language="bash")
        elif result is not None:
            model_used = result.get("Model_Used", selected_model_key)
            is_placeholder = bool(result.get("Is_Placeholder", not api_model_info.get(selected_model_key, {}).get("is_loaded", False)))
            if is_placeholder:
                st.markdown(
                    f"""
                    <div class="mode-card demo">
                        <h4>분석 모델: Placeholder 데모 응답</h4>
                        <p>FastAPI는 연결되어 있으나 선택 모델의 체크포인트가 로드되지 않았습니다. 아래 확률은 실제 학습 모델 출력이 아니라 시연용 응답입니다.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="mode-card live">
                        <h4>분석 모델: 실제 FastAPI 모델 추론</h4>
                        <p>{model_used} 체크포인트가 로드되어 업로드 이미지에 대한 실제 추론 결과를 표시합니다.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ── Top Disease 카드 ──────────────────────────────────────────
            top_disease = result["Top_Disease"].replace("_", " ")
            top_prob = result.get("Top_Probability", result.get(result["Top_Disease"], 0.0))
            st.markdown(f"""
            <div class="top-disease-card">
                <div class="sub">PRIMARY FINDING</div>
                <h2>{top_disease}</h2>
                <div class="prob">{top_prob:.1%}</div>
                <div class="sub">CONFIDENCE SCORE</div>
            </div>
            """, unsafe_allow_html=True)

            # ── 지표 행 ──────────────────────────────────────────────────
            m1, m2, m3 = st.columns(3)
            detected_count = len(result["Detected_Diseases"])
            with m1:
                st.markdown(f"""<div class="inference-metric"><div class="value">{result['Inference_Time_ms']}ms</div><div class="label">추론 시간</div></div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="inference-metric"><div class="value">{detected_count}</div><div class="label">감지된 질환</div></div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="inference-metric"><div class="value">14</div><div class="label">검사 질환</div></div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            probs = {label: result[label] for label in DISEASE_LABELS}
            case_id = result.get("Case_ID", f"CXR-{image_hash.upper()}")
            clinical_report = result.get("Clinical_Report", {}) or {}
            top_findings = clinical_report.get("Top_Findings_KR") or clinical_report.get("Top_Findings") or []
            top_findings_html = " ".join(
                f"<span class='disease-tag'>{escape(str(item))}</span>" for item in top_findings[:3]
            )

            # ── AI 판독문 초안 ───────────────────────────────────────────
            st.markdown('<div class="section-title">AI 판독문 초안</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="clinical-report-card">
                    <h4>복사·수정 가능한 판독 보조 초안</h4>
                    <p>
                        Case ID <b>{escape(case_id)}</b> · AI가 제시한 Findings/Impression 초안입니다.
                        최종 판독 전 원본 영상, Grad-CAM, 과거 영상, 임상 정보를 반드시 함께 확인하십시오.
                    </p>
                    <div style="margin-top:0.75rem;">{top_findings_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            edited_report = st.text_area(
                "판독문 초안 편집",
                value=result.get("Report_Draft") or clinical_report.get("Report_Draft_KR", ""),
                height=220,
                key=f"report_draft_{case_id}",
                help="의료진이 실제 판독문에 맞게 수정한 뒤 복사하거나 피드백 큐에 함께 저장할 수 있습니다.",
            )
            st.download_button(
                "판독문 초안 (.txt) 다운로드",
                data=edited_report.encode("utf-8"),
                file_name=f"{case_id}_ai_report_draft.txt",
                mime="text/plain",
                key=f"download_report_{case_id}",
            )
            if result.get("Need_Review_Reason"):
                st.markdown(
                    f"<div class='review-note'>검토 필요 사유: {escape(result.get('Need_Review_Reason', ''))}</div>",
                    unsafe_allow_html=True,
                )

            # ── 의료진 피드백 + 재학습 검수 큐 ───────────────────────────
            st.markdown('<div class="section-title">의료진 피드백 · 모델 재학습용 검수 큐</div>', unsafe_allow_html=True)
            queue_summary = get_feedback_queue_summary(limit=3) if health else None
            queue_count = queue_summary.get("total_count", 0) if queue_summary else 0
            st.markdown(
                f"""
                <div class="feedback-card">
                    <h4>판독의 검수 기록 저장</h4>
                    <p>
                        아래 버튼으로 AI 판단 동의/불일치, Grad-CAM 위치 오류, 라벨 수정, 코멘트를 저장합니다.
                        저장된 항목은 즉시 재학습하지 않고 <b>검수 큐</b>에 쌓아 병원별 라벨 정제와 모델 개선 후보로 활용합니다.
                    </p>
                    <span class="queue-badge">현재 검수 기록 {queue_count}건</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            reviewer_id = st.text_input(
                "판독의/검수자 ID 또는 이니셜",
                value="",
                placeholder="예: RAD01, Dr.Kim",
                key=f"reviewer_id_{case_id}",
            )
            corrected_labels = st.multiselect(
                "질환 라벨 수정이 필요한 경우 올바른 라벨을 선택하세요",
                options=DISEASE_LABELS,
                default=[],
                format_func=disease_display,
                key=f"corrected_labels_{case_id}",
            )
            clinician_comment = st.text_area(
                "판독의가 남긴 코멘트",
                placeholder="예: AI는 심비대를 높게 예측했지만 AP portable 촬영 영향으로 보이며 실제 판독은 정상 범위에 가깝습니다.",
                height=110,
                key=f"clinician_comment_{case_id}",
            )

            fb_cols = st.columns(5)
            for fb_col, feedback_type in zip(fb_cols, FEEDBACK_TYPES):
                with fb_col:
                    if st.button(feedback_type, key=f"fb_{feedback_type}_{case_id}", use_container_width=True):
                        if feedback_type == "질환 라벨 수정" and not corrected_labels:
                            st.warning("라벨 수정 피드백을 저장하려면 수정할 질환 라벨을 1개 이상 선택하세요.")
                        elif feedback_type == "판독의 코멘트" and not clinician_comment.strip():
                            st.warning("코멘트 저장을 위해 판독의 코멘트를 입력하세요.")
                        else:
                            submit_clinician_feedback(
                                feedback_type=feedback_type,
                                result=result,
                                probs=probs,
                                filename=filename,
                                threshold=threshold,
                                corrected_labels=corrected_labels,
                                comment=clinician_comment,
                                reviewer_id=reviewer_id,
                                edited_report=edited_report,
                            )

            if queue_summary and queue_summary.get("items"):
                with st.expander("최근 검수 큐 기록 보기"):
                    for item in queue_summary.get("items", []):
                        st.markdown(
                            f"- **{item.get('feedback_type', '-')}** · {item.get('case_id', '-')} · "
                            f"{item.get('submitted_at', '-')[:19]} · 상태: {item.get('review_status', '-')}"
                        )

            # ── 감지된 질환 태그 ──────────────────────────────────────────
            st.markdown('<div class="section-title">감지된 질환</div>', unsafe_allow_html=True)
            detected = [d for d in DISEASE_LABELS if result.get(d, 0) >= threshold]
            if detected:
                tags_html = " ".join(
                    f'<span class="disease-tag">{d.replace("_"," ")} ({result[d]:.0%})</span>'
                    for d in detected
                )
                st.markdown(f'<div class="premium-card">{tags_html}</div>', unsafe_allow_html=True)
            else:
                st.markdown("""<div class="premium-card" style="text-align:center;color:#047857;font-weight:750;">임계값 이상의 유의미한 질환이 감지되지 않았습니다.</div>""", unsafe_allow_html=True)

            # ── 질환 확률 차트 ────────────────────────────────────────────
            st.markdown('<div class="section-title">전체 질환 확률</div>', unsafe_allow_html=True)
            fig = create_disease_chart(probs, threshold)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
