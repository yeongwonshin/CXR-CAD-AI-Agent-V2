"""
📊 분석 결과 — CXR-CAD Analysis Dashboard.

라디오 버튼으로 지표별 화면을 전환하고,
각 화면 하단에 LangChain 기반 LLM 해석 / 질의응답 영역을 제공합니다.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from services.llm_analysis import (
    ask_metric_question,
    generate_metric_summary,
    get_configured_api_key,
    get_configured_model,
    langchain_is_ready,
    load_project_env,
)


# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CXR-CAD | 분석 결과",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    :root {
        --navy: #0f172a;
        --blue: #2563eb;
        --cyan: #06b6d4;
        --teal: #14b8a6;
        --violet: #7c3aed;
        --amber: #f59e0b;
        --line: rgba(148,163,184,0.25);
    }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 8% 8%, rgba(37,99,235,0.14), transparent 30%),
            radial-gradient(circle at 92% 12%, rgba(20,184,166,0.13), transparent 28%),
            linear-gradient(180deg, #f8fbff 0%, #eef7ff 48%, #f8fafc 100%);
    }
    .main .block-container { padding-top: 1.2rem; max-width: 1420px; }
    [data-testid="stSidebar"] {
        background:
            radial-gradient(circle at top left, rgba(56,189,248,0.22), transparent 30%),
            linear-gradient(180deg, #08111f 0%, #0f172a 52%, #111827 100%);
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
    [data-testid="stSidebar"] div[role="radiogroup"] label p,
    [data-testid="stSidebar"] div[role="radiogroup"] label span,
    [data-testid="stSidebar"] div[role="radiogroup"] div {
        color: #ffffff !important;
    }
    [data-testid="stSidebarNav"] span {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] hr { border-color: rgba(148,163,184,0.22) !important; }
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

    .main-header {
        background:
            radial-gradient(circle at 84% 20%, rgba(125,211,252,0.25), transparent 24%),
            linear-gradient(135deg, #08111f 0%, #15345f 54%, #0f766e 100%);
        border: 1px solid rgba(125,211,252,0.24);
        border-radius: 28px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 24px 58px rgba(15,23,42,0.20);
    }
    .main-header .eyebrow { color:#a7f3d0 !important; font-size:0.72rem; letter-spacing:0.16em; text-transform:uppercase; font-weight:800; margin-bottom:0.35rem; }
    .main-header h1 { margin:0; font-size:1.72rem; font-weight:850; color:white !important; letter-spacing:-0.03em; }
    .main-header p  { margin:0.35rem 0 0; font-size:0.9rem; opacity:0.9; color:#dbeafe !important; line-height:1.5; }

    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.96) 0%, rgba(239,246,255,0.9) 100%);
        border: 1px solid rgba(125,211,252,0.36);
        border-radius: 20px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 16px 34px rgba(15,23,42,0.08);
        margin-bottom: 0.8rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-card .value { font-size: 2.05rem; font-weight: 850; margin: 0; letter-spacing:-0.03em; }
    .metric-card .label { font-size: 0.74rem; text-transform: uppercase; letter-spacing:1.3px; color: #64748b; margin-top: 0.25rem; font-weight:750; }

    .section-header {
        font-size: 1.12rem; font-weight: 850; color: #0f172a;
        margin: 1.5rem 0 0.75rem;
        padding-bottom: 0.58rem;
        border-bottom: 1px solid rgba(148,163,184,0.28);
        display:flex; align-items:center; gap:0.5rem;
    }
    .section-header::before {
        content:""; width:0.55rem; height:0.55rem; border-radius:999px;
        background: linear-gradient(135deg, var(--blue), var(--teal));
        box-shadow: 0 0 0 5px rgba(14,165,233,0.12);
    }
    .analysis-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(148,163,184,0.24);
        border-radius: 22px;
        padding: 1.5rem;
        box-shadow: 0 18px 42px rgba(15,23,42,0.08);
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    .insight-box {
        background: linear-gradient(135deg, #eff6ff, #ecfeff);
        border: 1px solid rgba(125,211,252,0.55);
        border-radius: 16px;
        padding: 1rem 1.15rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #0f4c81;
        box-shadow: 0 12px 26px rgba(14,165,233,0.08);
    }
    .warning-box {
        background: linear-gradient(135deg, #fff7ed, #fef3c7);
        border: 1px solid rgba(245,158,11,0.45);
        border-radius: 16px;
        padding: 1rem 1.15rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #78350f;
        box-shadow: 0 12px 26px rgba(245,158,11,0.10);
    }
    .llm-box {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
        border: 1px solid rgba(167,139,250,0.38);
        border-radius: 20px;
        padding: 1.2rem 1.3rem;
        margin-top: 0.6rem;
        box-shadow: 0 18px 42px rgba(15,23,42,0.06);
    }
    .sidebar-brand {
        border: 1px solid rgba(125,211,252,0.22);
        border-radius: 18px;
        padding: 1rem;
        background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,58,138,0.5));
        box-shadow: 0 18px 48px rgba(8,17,31,0.35);
    }
    .sidebar-brand h2 { margin:0; font-size:1.12rem; color:white !important; font-weight:850; }
    .sidebar-brand p { margin:0.32rem 0 0; color:#b6c7dc !important; font-size:0.78rem; line-height:1.45; }

    #MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ── 상수 ──────────────────────────────────────────────────────────────────────
load_project_env()

BASE_CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))
DEFAULT_LLM_MODEL  = get_configured_model("gpt-4o-mini")

SUPPORTED_MODELS = {
    "densenet":    "DenseNet-121",
    "efficientnet": "EfficientNet-B4",
    "vit":         "ViT-B/16",
}


# ── 데이터 로드 함수 ────────────────────────────────────────────────────────
def load_csv_data(checkpoint_dir: Path, filename: str, fallback_cols: list) -> pd.DataFrame:
    path = checkpoint_dir / filename
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.error(f"Failed to read {filename}: {e}")
    return pd.DataFrame(columns=fallback_cols)


def load_model_data(checkpoint_dir: Path) -> dict:
    """선택된 모델의 서브디렉토리에서 결과 CSV를 모두 로드."""
    return {
        "op":     load_csv_data(checkpoint_dir, "op_analysis.csv",
                                ["기준", "Threshold", "Sensitivity", "Specificity", "PPV", "NPV"]),
        "gender": load_csv_data(checkpoint_dir, "gender_subgroup.csv",
                                ["Disease", "Male AUROC", "Female AUROC", "Gap"]),
        "age":    load_csv_data(checkpoint_dir, "age_subgroup.csv",
                                ["Age Group", "N", "Mean AUROC"]),
        "view":   load_csv_data(checkpoint_dir, "view_subgroup.csv",
                                ["View", "N", "Mean AUROC", "Gap vs PA"]),
        "ext":    load_csv_data(checkpoint_dir, "domain_shift.csv",
                                ["Disease", "NIH AUROC", "CheXpert AUROC", "Gap"]),
        "fp":     load_csv_data(checkpoint_dir, "false_positive.csv",
                                ["Case", "예측", "GT", "확률", "Grad-CAM", "원인"]),
        "fn":     load_csv_data(checkpoint_dir, "false_negative.csv",
                                ["Case", "예측", "GT", "확률", "Grad-CAM", "원인"]),
        "region": load_csv_data(checkpoint_dir, "shortcut_regions.csv",
                                ["영역", "Count"]),
    }


# ── 모델 선택 (session_state 기반, 사이드바보다 먼저 실행) ────────────────────
if "analysis_llm_enabled" not in st.session_state:
    st.session_state["analysis_llm_enabled"] = True

_selected_model = st.session_state.get("analysis_selected_model", "densenet")
CHECKPOINT_DIR = BASE_CHECKPOINT_DIR / _selected_model
_data = load_model_data(CHECKPOINT_DIR)

# 렌더 함수들이 참조하는 전역 변수 (모델 전환 시 자동 갱신)
EXAMPLE_OP        = _data["op"]
EXAMPLE_GENDER    = _data["gender"]
EXAMPLE_AGE       = _data["age"]
EXAMPLE_VIEW      = _data["view"]
EXAMPLE_EXT       = _data["ext"]
FALSE_POSITIVE_DF = _data["fp"]
FALSE_NEGATIVE_DF = _data["fn"]
REGION_DF         = _data["region"]


# ── 차트 헬퍼 ─────────────────────────────────────────────────────────────────
def dynamic_yaxis_range(*series: pd.Series | list, pad: float = 0.04) -> list[float]:
    """Return a visible [min, max] range for AUROC/probability charts."""
    values: list[float] = []
    for s in series:
        vals = pd.to_numeric(pd.Series(s), errors="coerce").dropna().astype(float).tolist()
        values.extend(vals)
    if not values:
        return [0.0, 1.0]
    lo, hi = min(values), max(values)
    if lo == hi:
        lo -= pad
        hi += pad
    else:
        span = hi - lo
        lo -= max(pad, span * 0.18)
        hi += max(pad, span * 0.18)
    return [max(0.0, lo), min(1.0, hi)]


def chart_operating_point(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, color in [
        ("Sensitivity", "#3b82f6"),
        ("Specificity", "#10b981"),
        ("PPV", "#f59e0b"),
        ("NPV", "#8b5cf6"),
    ]:
        fig.add_trace(
            go.Bar(
                x=df["기준"],
                y=df[col],
                name=col,
                marker=dict(color=color),
                text=[f"{v:.3f}" for v in df[col]],
                textposition="outside",
            )
        )
    fig.update_layout(
        height=380,
        barmode="group",
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Operating Point Analysis (Cardiomegaly)", font=dict(size=14, family="Inter")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        yaxis=dict(range=[0, 1.12]),
    )
    return fig



def chart_subgroup_gender(df: pd.DataFrame) -> go.Figure:
    d = df[df["Disease"] != "Mean"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=d["Disease"], y=d["Male AUROC"], name="Male", marker=dict(color="#3b82f6"))
    )
    fig.add_trace(
        go.Bar(x=d["Disease"], y=d["Female AUROC"], name="Female", marker=dict(color="#ec4899"))
    )
    fig.update_layout(
        height=340,
        barmode="group",
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Subgroup: Gender AUROC", font=dict(size=14, family="Inter")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        yaxis=dict(range=dynamic_yaxis_range(d["Male AUROC"], d["Female AUROC"])),
    )
    return fig



def chart_subgroup_age(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=df["Age Group"],
            y=df["Mean AUROC"],
            marker=dict(color=["#f59e0b", "#10b981", "#6366f1"]),
            text=[f"{v:.4f}" for v in df["Mean AUROC"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Subgroup: Age Group AUROC", font=dict(size=14, family="Inter")),
        yaxis=dict(range=dynamic_yaxis_range(df["Mean AUROC"])),
    )
    return fig



def chart_subgroup_view(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=df["View"],
            y=df["Mean AUROC"],
            marker=dict(color=["#3b82f6", "#ef4444"]),
            text=[f"{v:.4f}" for v in df["Mean AUROC"]],
            textposition="outside",
            width=0.4,
        )
    )
    fig.add_hline(y=0.8, line_dash="dot", line_color="#94a3b8", annotation_text="Baseline 0.80")
    fig.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Subgroup: View Position (PA vs AP)", font=dict(size=14, family="Inter")),
        yaxis=dict(range=dynamic_yaxis_range(list(df["Mean AUROC"]) + [0.8])),
    )
    return fig



def chart_external_val(df: pd.DataFrame) -> go.Figure:
    d = df[df["Disease"] != "Mean"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=d["Disease"], y=d["NIH AUROC"], name="NIH (Internal)", marker=dict(color="#3b82f6"))
    )
    fig.add_trace(
        go.Bar(
            x=d["Disease"],
            y=d["CheXpert AUROC"],
            name="CheXpert (External)",
            marker=dict(color="#f97316"),
        )
    )
    fig.update_layout(
        height=380,
        barmode="group",
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="External Validation: NIH vs CheXpert", font=dict(size=14, family="Inter")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        yaxis=dict(range=[0.6, 1.0]),
    )
    return fig



def chart_domain_gap(df: pd.DataFrame) -> go.Figure:
    d = df.copy()
    d["gap_numeric"] = d["CheXpert AUROC"] - d["NIH AUROC"]
    colors = ["#ef4444" if g < 0 else "#10b981" for g in d["gap_numeric"]]
    fig = go.Figure(
        go.Bar(
            x=d["Disease"],
            y=d["gap_numeric"],
            marker=dict(color=colors),
            text=[f"{v:+.1%}" for v in d["gap_numeric"]],
            textposition="outside",
        )
    )
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Domain Shift Gap (CheXpert − NIH)", font=dict(size=14, family="Inter")),
        yaxis=dict(title="AUROC Gap"),
    )
    return fig



def chart_region_shift(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Pie(
            labels=df["영역"],
            values=df["Count"],
            hole=0.55,
            marker=dict(colors=["#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#94a3b8"]),
            textinfo="label+percent",
            textfont=dict(size=12, family="Inter"),
        )
    )
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(text="Grad-CAM 활성화 영역 분포 (100건)", font=dict(size=14, family="Inter")),
    )
    return fig


# ── 공통 헬퍼 ─────────────────────────────────────────────────────────────────
def to_context_block(name: str, df: pd.DataFrame) -> str:
    return f"[{name}]\n{df.to_csv(index=False)}"



def metric_card(label: str, value: str, color: str = "#0f172a") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <p class="value" style="color:{color};">{value}</p>
            <p class="label">{label}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )



def heuristic_summary(metric_key: str) -> str:
    summaries = {
        "operating_point": (
            "- Sensitivity 90% 기준은 놓치는 환자를 줄이는 대신 PPV가 0.081로 낮아 추가 검사 부담이 커집니다.\n"
            "- Specificity 90% 기준은 불필요한 후속 조치를 줄이지만 sensitivity가 0.689로 내려갑니다.\n"
            "- 따라서 스크리닝과 확진 보조를 같은 threshold로 운영하면 임상 목적이 충돌할 수 있습니다."
        ),
        "gender": (
            "- 평균 AUROC 차이는 작지만 질환별로 편차 방향이 다릅니다.\n"
            "- Cardiomegaly는 남성 쪽이 높고, Effusion/Hernia는 여성 쪽이 더 높습니다.\n"
            "- 따라서 전체 평균만 보면 subgroup 편향을 놓칠 수 있습니다."
        ),
        "age": (
            "- 40-60세 구간에서 AUROC가 가장 높고 표본 수도 가장 많습니다.\n"
            "- 60+ 구간은 성능이 소폭 하락해 고령군 일반화 점검이 필요합니다.\n"
            "- 표본 수와 질환 복잡도가 함께 성능 차이에 영향을 준 것으로 볼 수 있습니다."
        ),
        "view": (
            "- AP 영상 AUROC가 PA 대비 5.2% 낮아 촬영 조건 차이가 성능에 영향을 줍니다.\n"
            "- 응급/이동식 촬영 비중이 높은 AP에서 domain shift 가능성이 큽니다.\n"
            "- View별 augmentation 또는 분리 보정 전략이 우선 후보입니다."
        ),
        "external_validation": (
            "- 내부 NIH 대비 외부 CheXpert 성능이 전반적으로 하락합니다.\n"
            "- 특히 Pneumonia에서 하락폭이 가장 커 외부 일반화 리스크가 큽니다.\n"
            "- 외부 데이터 재보정과 threshold 재설정이 필요합니다."
        ),
        "domain_gap": (
            "- 모든 주요 질환에서 외부 데이터 성능이 음의 gap을 보입니다.\n"
            "- 이는 촬영기관, 라벨링, 환자군 차이의 누적 영향으로 해석할 수 있습니다.\n"
            "- 배포 전 site-specific calibration 없이는 실제 운영 성능 저하 가능성이 큽니다."
        ),
        "error_cases": (
            "- False positive는 쇄골, 유방 그림자, 혈관 단면처럼 구조적 혼동이 많습니다.\n"
            "- False negative는 작은 병변과 diffuse pattern에서 집중 실패가 두드러집니다.\n"
            "- 따라서 hard negative mining과 작은 병변 증강이 우선 과제입니다."
        ),
        "region_shift": (
            "- 13%가 폐 외 영역(의료기기+텍스트/마커)에 반응해 shortcut learning 신호가 있습니다.\n"
            "- 모델이 병변 자체보다 촬영 문맥에 의존할 위험이 있습니다.\n"
            "- 폐 영역 마스킹과 attention 제약을 검토할 필요가 있습니다."
        ),
    }
    return summaries.get(metric_key, "- 현재 지표에 대한 규칙 기반 요약이 없습니다.")



def render_llm_section(metric_key: str, metric_title: str, metric_context: str) -> None:
    st.markdown('<div class="section-header">LLM 결론 및 자료 해석</div>', unsafe_allow_html=True)

    if not st.session_state.get("analysis_llm_enabled", True):
        st.info("사이드바에서 LLM 사용이 꺼져 있어 이 화면은 규칙 기반 고속 해석만 표시합니다. 지표를 1→2→3으로 이동해도 이 설정은 유지됩니다.")
        st.markdown('<div class="llm-box">', unsafe_allow_html=True)
        st.markdown("### 고속 해석\n" + heuristic_summary(metric_key))
        st.markdown('</div>', unsafe_allow_html=True)
        return

    api_key = get_configured_api_key()
    model_name = get_configured_model(DEFAULT_LLM_MODEL)
    ready, import_error = langchain_is_ready()

    summary_cache = st.session_state.setdefault("analysis_llm_summary_cache", {})
    summary_slot = st.container()
    cache_key = f"{metric_key}::{model_name}"

    if ready and api_key:
        if cache_key not in summary_cache:
            with summary_slot:
                with st.spinner("LLM이 압축 지표 컨텍스트로 빠르게 분석 중입니다..."):
                    try:
                        summary_cache[cache_key] = generate_metric_summary(
                            metric_title=metric_title,
                            metric_context=metric_context,
                            api_key=api_key,
                            model_name=model_name,
                        )
                    except Exception as exc:
                        summary_cache[cache_key] = f"LLM 요약 생성 실패: {exc}"

        with summary_slot:
            st.caption(f"모델: {model_name}")
            st.markdown('<div class="llm-box">', unsafe_allow_html=True)
            st.markdown(summary_cache[cache_key])
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        with summary_slot:
            if not ready:
                st.info(f"LangChain/OpenAI 패키지가 아직 설치되지 않았습니다: {import_error}")
            elif not api_key:
                st.info("프로젝트 루트의 .env에 OPENAI_API_KEY를 설정하면 LLM 해석이 자동 생성됩니다.")
            st.markdown('<div class="llm-box">', unsafe_allow_html=True)
            st.markdown("### 임시 해석\n" + heuristic_summary(metric_key))
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">LLM에 질문하기</div>', unsafe_allow_html=True)
    with st.form(key=f"qa_form_{metric_key}"):
        question = st.text_area(
            "현재 화면의 지표에 대해 질문하세요.",
            placeholder="예: AP 촬영에서 성능이 낮은 가장 큰 원인은 무엇으로 보이나요?",
            key=f"qa_input_{metric_key}",
            height=110,
        )
        submitted = st.form_submit_button("질문 보내기")

    if submitted:
        if not question.strip():
            st.warning("질문을 입력해 주세요.")
        elif not ready:
            st.warning("질문 응답을 사용하려면 LangChain/OpenAI 패키지 설치가 필요합니다.")
        elif not api_key:
            st.warning("질문 응답을 사용하려면 프로젝트 루트의 .env에 OPENAI_API_KEY를 설정해 주세요.")
        else:
            with st.spinner("LLM이 질문에 답하는 중입니다..."):
                try:
                    answer = ask_metric_question(
                        metric_title=metric_title,
                        metric_context=metric_context,
                        question=question.strip(),
                        api_key=api_key,
                        model_name=model_name,
                    )
                    st.session_state[f"qa_answer::{metric_key}"] = answer
                    st.session_state[f"qa_question::{metric_key}"] = question.strip()
                except Exception as exc:
                    st.session_state[f"qa_answer::{metric_key}"] = f"질문 응답 생성 실패: {exc}"
                    st.session_state[f"qa_question::{metric_key}"] = question.strip()

    saved_answer = st.session_state.get(f"qa_answer::{metric_key}")
    saved_question = st.session_state.get(f"qa_question::{metric_key}")
    if saved_answer:
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        st.markdown(f"**질문**\n\n{saved_question}")
        st.markdown("**답변**")
        st.markdown(saved_answer)
        st.markdown('</div>', unsafe_allow_html=True)


# ── 화면별 렌더 함수 ───────────────────────────────────────────────────────────
def render_operating_point() -> str:
    st.markdown('<div class="section-header">1. Operating Point 분석 (Cardiomegaly)</div>', unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    with k1:
        metric_card("권장 스크리닝 Threshold", f"{EXAMPLE_OP.loc[1, 'Threshold']:.2f}", "#2563eb")
    with k2:
        metric_card("권장 확진보조 Threshold", f"{EXAMPLE_OP.loc[2, 'Threshold']:.2f}", "#16a34a")
    with k3:
        metric_card("최고 NPV", f"{EXAMPLE_OP['NPV'].max():.3f}", "#7c3aed")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(chart_operating_point(EXAMPLE_OP), width="stretch", config={"displayModeBar": False})
    with c2:
        st.dataframe(EXAMPLE_OP, hide_index=True, width="stretch")
        tab_screen, tab_confirm = st.tabs(["스크리닝", "확진 보조"])
        with tab_screen:
            st.markdown(
                """
                **권장:** Sensitivity 90% (Threshold=0.28)  
                위음성(놓치는 환자) 최소화 최우선  
                Trade-off: False Positive 증가 → 추가 검사 비용 상승
                """
            )
        with tab_confirm:
            st.markdown(
                """
                **권장:** Specificity 90% (Threshold=0.56)  
                불필요한 추가 검사/환자 불안 최소화  
                Trade-off: 일부 양성 케이스 누락 가능
                """
            )
    return (
        "Metric focus: Operating Point analysis for Cardiomegaly.\n"
        + to_context_block("Operating Point Table", EXAMPLE_OP)
        + "\nClinical note: screening should prioritize sensitivity, confirmatory support should prioritize specificity."
    )



def render_gender() -> str:
    st.markdown('<div class="section-header">2. Subgroup Analysis — Gender</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(chart_subgroup_gender(EXAMPLE_GENDER), width="stretch", config={"displayModeBar": False})
    with c2:
        st.dataframe(EXAMPLE_GENDER, hide_index=True, width="stretch")
        st.markdown(
            """
            <div class="insight-box">
            전체 평균 차이는 작지만 질환별로 편차 방향이 다릅니다.<br>
            평균값만 보면 특정 질환에서의 subgroup 불균형을 놓칠 수 있습니다.
            </div>
            """,
            unsafe_allow_html=True,
        )
    return "Metric focus: subgroup analysis by gender.\n" + to_context_block("Gender AUROC", EXAMPLE_GENDER)



def render_age() -> str:
    st.markdown('<div class="section-header">3. Subgroup Analysis — Age Group</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(chart_subgroup_age(EXAMPLE_AGE), width="stretch", config={"displayModeBar": False})
    with c2:
        st.dataframe(EXAMPLE_AGE, hide_index=True, width="stretch")
        st.markdown(
            """
            <div class="insight-box">
            40-60세가 최다 학습 데이터 → 최적 성능<br>
            60+ 그룹은 동반질환 복잡성으로 성능 하락 가능성
            </div>
            """,
            unsafe_allow_html=True,
        )
    return "Metric focus: subgroup analysis by age group.\n" + to_context_block("Age Group AUROC", EXAMPLE_AGE)



def render_view() -> str:
    st.markdown('<div class="section-header">4. Subgroup Analysis — View Position</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(chart_subgroup_view(EXAMPLE_VIEW), width="stretch", config={"displayModeBar": False})
    with c2:
        st.dataframe(EXAMPLE_VIEW, hide_index=True, width="stretch")
        st.markdown(
            """
            <div class="warning-box">
            <b>PA/AP 간 성능 차이 5.2%</b><br>
            AP는 이동식 응급 촬영이 많아 영상 품질과 분포가 다를 수 있습니다.<br>
            <b>권장:</b> AP 영상 별도 증강 또는 도메인 적응 적용
            </div>
            """,
            unsafe_allow_html=True,
        )
    return "Metric focus: subgroup analysis by view position.\n" + to_context_block("View Position AUROC", EXAMPLE_VIEW)



def render_external_validation() -> str:
    st.markdown('<div class="section-header">5. External Validation (CheXpert)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_external_val(EXAMPLE_EXT), width="stretch", config={"displayModeBar": False})
    with c2:
        st.dataframe(EXAMPLE_EXT, hide_index=True, width="stretch")

    info_cols = st.columns(3)
    with info_cols[0]:
        st.markdown(
            """
            <div class="warning-box">
            <b>촬영 기관</b><br>NIH: 30개 다기관<br>CheXpert: Stanford 단일 기관
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info_cols[1]:
        st.markdown(
            """
            <div class="warning-box">
            <b>라벨링 방식</b><br>NIH: NLP 자동<br>CheXpert: 전문의 검토
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info_cols[2]:
        st.markdown(
            """
            <div class="warning-box">
            <b>환자군</b><br>NIH: 외래 중심<br>CheXpert: 입원 포함, 중증도↑
            </div>
            """,
            unsafe_allow_html=True,
        )
    return "Metric focus: external validation between NIH and CheXpert.\n" + to_context_block("External Validation", EXAMPLE_EXT)



def render_domain_gap() -> str:
    st.markdown('<div class="section-header">6. Domain Shift Gap</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_domain_gap(EXAMPLE_EXT), width="stretch", config={"displayModeBar": False})
    st.dataframe(EXAMPLE_EXT[["Disease", "Gap"]], hide_index=True, width="stretch")
    st.markdown(
        """
        <div class="warning-box">
        외부 데이터셋으로 갈수록 모든 주요 질환에서 음의 gap이 나타납니다.<br>
        배포 전 site-specific calibration과 threshold 재점검이 필요합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )
    return "Metric focus: domain shift gap between internal and external validation.\n" + to_context_block("Domain Gap", EXAMPLE_EXT)



def render_error_cases() -> str:
    st.markdown('<div class="section-header">7. Error Analysis — False Positive / False Negative</div>', unsafe_allow_html=True)
    tab_fp, tab_fn = st.tabs(["False Positive Top 5", "False Negative Top 5"])
    with tab_fp:
        st.dataframe(FALSE_POSITIVE_DF, hide_index=True, width="stretch")
    with tab_fn:
        st.dataframe(FALSE_NEGATIVE_DF, hide_index=True, width="stretch")
    return (
        "Metric focus: top false positive and false negative cases.\n"
        + to_context_block("False Positives", FALSE_POSITIVE_DF)
        + "\n"
        + to_context_block("False Negatives", FALSE_NEGATIVE_DF)
    )



def render_region_shift() -> str:
    st.markdown('<div class="section-header">8. Error Analysis — 폐 영역 이탈 분석</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_region_shift(REGION_DF), width="stretch", config={"displayModeBar": False})
    st.markdown(
        """
        <div class="warning-box">
        <b>Shortcut Learning 의심:</b> 의료기기(8건) + 텍스트/마커(5건) = 13건<br>
        <b>개선 방향:</b> 폐 영역 마스킹, attention 제약, artifact suppression 검토
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(REGION_DF, hide_index=True, width="stretch")
    return "Metric focus: region shift and shortcut learning evidence.\n" + to_context_block("Region Distribution", REGION_DF)


METRIC_PAGES: OrderedDict[str, dict[str, str | Callable[[], str]]] = OrderedDict(
    {
        "operating_point": {"label": "1. Operating Point", "title": "Operating Point 분석", "render": render_operating_point},
        "gender": {"label": "2. Gender", "title": "Subgroup Analysis - Gender", "render": render_gender},
        "age": {"label": "3. Age Group", "title": "Subgroup Analysis - Age Group", "render": render_age},
        "view": {"label": "4. View Position", "title": "Subgroup Analysis - View Position", "render": render_view},
        "external_validation": {"label": "5. External Validation", "title": "External Validation", "render": render_external_validation},
        "domain_gap": {"label": "6. Domain Shift Gap", "title": "Domain Shift Gap", "render": render_domain_gap},
        "error_cases": {"label": "7. Error Cases", "title": "Error Analysis - FP/FN", "render": render_error_cases},
        "region_shift": {"label": "8. Region Shift", "title": "Error Analysis - Region Shift", "render": render_region_shift},
    }
)


# ── 헤더 ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="main-header">
    <div class="eyebrow">Analysis results</div>
    <h1>CXR-CAD — 분석 결과 대시보드</h1>
    <p>지표별 화면 전환 · 모델 성능 비교 · LLM 기반 자료 해석 및 질의응답</p>
</div>
""",
    unsafe_allow_html=True,
)


# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div class="sidebar-brand"><h2>분석 결과</h2><p>지표별 분석 화면과 LLM 해석을 제공합니다.</p></div>""", unsafe_allow_html=True)
    st.divider()

    # ── 모델 선택기 ──────────────────────────────────────────────────────────
    st.markdown("### 분석 모델")
    st.selectbox(
        "결과를 볼 모델 선택",
        options=list(SUPPORTED_MODELS.keys()),
        format_func=lambda k: SUPPORTED_MODELS[k],
        key="analysis_selected_model",
    )
    st.caption(f"Data path: `checkpoints/{st.session_state['analysis_selected_model']}/`")

    has_real = any(
        (CHECKPOINT_DIR / f).exists()
        for f in ["test_predictions.csv", "op_analysis.csv"]
    )
    if has_real:
        st.success("실제 결과 데이터가 감지되었습니다.")
    else:
        st.info("결과 없음 — 학습 후 체크포인트를 배치하세요.")

    st.divider()
    selected_metric = st.radio(
        "표시할 분석 항목",
        options=list(METRIC_PAGES.keys()),
        format_func=lambda key: str(METRIC_PAGES[key]["label"]),
    )

    st.divider()
    st.markdown("### LLM 사용")
    st.toggle(
        "Result Analysis에서 LLM 사용",
        key="analysis_llm_enabled",
        help="끄면 지표 화면을 1→2→3으로 이동해도 LLM 호출이 다시 발생하지 않고, 이 토글 상태도 유지됩니다.",
    )

    if st.session_state.get("analysis_llm_enabled", True):
        ready, import_error = langchain_is_ready()
        configured_key = bool(get_configured_api_key())
        configured_model = get_configured_model(DEFAULT_LLM_MODEL)
        if ready and configured_key:
            st.success(f".env 기반 LLM 설정 감지됨 · 모델: {configured_model}")
            st.caption("고속 모드: LangChain 없이 OpenAI-compatible 직접 호출")
        elif ready:
            st.info(".env의 OPENAI_API_KEY가 비어 있습니다. LLM 대신 규칙 기반 해석이 표시됩니다.")
        else:
            st.warning(f"LLM 클라이언트 준비 실패: {import_error}")
    else:
        st.info("LLM OFF · 지표 전환 시 LLM 재호출 없음")

    st.divider()
    st.markdown(
        "<div style='text-align:center;opacity:0.45;font-size:0.72rem;'>"
        "CXR-CAD v0.3.1<br>Fast Analysis + Optional LLM"
        "</div>",
        unsafe_allow_html=True,
    )



# ── 본문 렌더 ─────────────────────────────────────────────────────────────────
page_conf = METRIC_PAGES[selected_metric]
render_fn = page_conf["render"]
metric_title = str(page_conf["title"])

st.caption(f"현재 화면: {page_conf['label']}")
metric_context = render_fn()
render_llm_section(selected_metric, metric_title, metric_context)
