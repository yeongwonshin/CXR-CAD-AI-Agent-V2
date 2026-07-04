"""
CXR-CAD Backend API Service.

FastAPI 엔드포인트:
  GET  /health   → 서비스 상태 확인 (모델 로드 여부 포함)
  GET  /models   → 지원 모델 목록
  POST /predict  → 흉부 X-ray 분석 (PNG/JPEG/DICOM)

가중치 로드 규칙:
  - 서버 시작 시 checkpoints/<model_key>/<model_key>_best.pth 자동 탐색
  - 파일이 존재하면 실제 모델 추론, 없으면 Placeholder 모드
  - .pth 파일은 절대 Git 저장소에 포함하지 않습니다 (.gitignore 참조)

체크포인트 저장 포맷 (Colab 학습 코드와 호환):
    torch.save({
        "epoch"              : epoch,
        "model_state_dict"   : model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_auroc"          : best_auroc,
    }, "checkpoints/<model_key>/<model_key>_best.pth")
"""

from __future__ import annotations

import io
import json
import os
import random
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from api.schemas import (
    AgentBatchResponse,
    AgentCaseResult,
    AgentChatRequest,
    AgentChatResponse,
    FeedbackQueueResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResult,
)
from src.preprocess.dicom_utils import dicom_to_pil, is_dicom, parse_dicom_metadata
from src.preprocess.transforms import preprocess_single_image
from src.agentic import build_agent_batch_summary, generate_llm_agent_reply, get_agent_runtime_status
from src.agentic.cxr_agent import analyze_image_quality, build_anatomy_assessment, build_triage_assessment
from src.agentic.dynamic_agent import CXRCaseState, CXRRuntimeTool, DynamicBatchWorkflowAgent, DynamicCXRWorkflowAgent
from src.train.models import (
    DISEASE_LABELS,
    SUPPORTED_MODELS,
    build_model,
    get_model_info,
)

# ── 설정 ─────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR      = Path(os.getenv("CHECKPOINT_DIR", "checkpoints"))
FEEDBACK_QUEUE_PATH = Path(os.getenv("FEEDBACK_QUEUE_PATH", "data/feedback_queue.jsonl"))
DETECTION_THRESHOLD = 0.3
API_VERSION         = "0.2.0"
DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Placeholder Grad-CAM (1×1 빨간 픽셀 PNG, Base64)
_FAKE_GRADCAM_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8BQDwAEgAF/"
    "poIuwwAAAABJRU5ErkJggg=="
)


# ── 모델 레지스트리 ───────────────────────────────────────────────────────────

# { model_key: nn.Module | None }  None = 체크포인트 없음 (Placeholder 모드)
_model_registry: Dict[str, Optional[object]] = {k: None for k in SUPPORTED_MODELS}


def _find_checkpoint(model_key: str) -> Optional[Path]:
    """
    checkpoints/<model_key>/ 서브디렉토리에서 <model_key>_best.pth 탐색.

    탐색 우선순위:
      1. checkpoints/<model_key>/<model_key>_best.pth  ← 신규 구조
      2. checkpoints/<model_key>/*.pth (glob)           ← 신규 구조 변형
      3. checkpoints/<model_key>_best.pth              ← 구버전 flat 구조 (하위 호환)
    ex) checkpoints/densenet/densenet_best.pth
    """
    if not CHECKPOINT_DIR.exists():
        return None
    # 1) 신규: 서브디렉토리 내 직접 매칭
    sub_direct = CHECKPOINT_DIR / model_key / f"{model_key}_best.pth"
    if sub_direct.exists():
        return sub_direct
    # 2) 신규: 서브디렉토리 내 glob
    sub_candidates = sorted((CHECKPOINT_DIR / model_key).glob(f"{model_key}*.pth"), reverse=True)
    if sub_candidates:
        return sub_candidates[0]
    # 3) 구버전 flat 구조 fallback
    direct = CHECKPOINT_DIR / f"{model_key}_best.pth"
    if direct.exists():
        return direct
    candidates = sorted(CHECKPOINT_DIR.glob(f"{model_key}*.pth"), reverse=True)
    return candidates[0] if candidates else None


def _load_checkpoint_weights(model_key: str, ckpt_path: Path) -> bool:
    """
    .pth 파일에서 모델 가중치를 로드합니다.

    지원 state_dict 키 포맷:
      - {"model_state_dict": ...}   ← Colab 학습 표준 포맷
      - {"state_dict": ...}
      - 직접 state_dict (dict of tensors)

    Returns:
        True: 로드 성공, False: 실패
    """
    try:
        model = build_model(model_key)
        ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)

        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        else:
            raise ValueError("알 수 없는 체크포인트 포맷")

        model.load_state_dict(state_dict, strict=True)
        model.to(DEVICE)
        model.eval()
        _model_registry[model_key] = model

        val_auroc = ckpt.get("val_auroc", "N/A") if isinstance(ckpt, dict) else "N/A"
        print(f"  ✅ [{model_key}] {ckpt_path.name} 로드 완료 (val_auroc={val_auroc})")
        return True

    except Exception as e:
        print(f"  ⚠️  [{model_key}] 로드 실패: {e}")
        return False


# ── Lifespan (서버 시작/종료 훅) ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    서버 시작 시 checkpoints/<model>/ 폴더의 .pth 파일을 자동으로 탐색·로드합니다.

    - .pth 없음   → Placeholder 모드 (시뮬레이션 예측값 반환)
    - .pth 있음   → 실제 모델 추론
    """
    print(f"\n🩺 CXR-CAD API v{API_VERSION} 시작")
    print(f"   Device    : {DEVICE}")
    print(f"   Checkpoint: {CHECKPOINT_DIR.resolve()}")
    print("   모델 가중치 탐색 중...")

    loaded_any = False
    for key in SUPPORTED_MODELS:
        ckpt = _find_checkpoint(key)
        if ckpt:
            if _load_checkpoint_weights(key, ckpt):
                loaded_any = True
        else:
            print(f"  ℹ️  [{key}] 체크포인트 없음 → Placeholder 모드")

    if not loaded_any:
        print("\n   ⚠️  모든 모델이 Placeholder 모드로 동작합니다.")
        print("   Colab에서 학습 후 .pth 파일을 checkpoints/<model>/ 에 저장하세요.\n")

    app.state.loaded_models = [k for k, v in _model_registry.items() if v is not None]
    yield
    print("🩺 CXR-CAD API 종료")


# ── FastAPI App ───────────────────────────────────────────────────────────────

API_MODELS = SUPPORTED_MODELS + ["ensemble"]

app = FastAPI(
    title="CXR-CAD API",
    description=(
        "흉부 X-ray 컴퓨터 보조 진단 API.\n\n"
        "**지원 모델**: Ensemble, DenseNet-121, EfficientNet-B4, ViT-B/16\n\n"
        "모델 가중치는 Colab 학습 후 `checkpoints/<model>/` 서브폴더에 `.pth` 파일로 배치합니다.\n"
        "가중치 파일이 없으면 Placeholder 모드로 동작합니다."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Placeholder 예측 (가중치 없을 때) ────────────────────────────────────────

_PLACEHOLDER_BASE: Dict[str, Dict[str, float]] = {
    "densenet": {
        "Atelectasis": 0.32, "Cardiomegaly": 0.85, "Effusion": 0.50,
        "Infiltration": 0.18, "Mass": 0.12, "Nodule": 0.08,
        "Pneumonia": 0.22, "Pneumothorax": 0.05, "Consolidation": 0.15,
        "Edema": 0.42, "Emphysema": 0.03, "Fibrosis": 0.07,
        "Pleural_Thickening": 0.11, "Hernia": 0.02,
    },
    "efficientnet": {
        "Atelectasis": 0.29, "Cardiomegaly": 0.88, "Effusion": 0.53,
        "Infiltration": 0.20, "Mass": 0.14, "Nodule": 0.09,
        "Pneumonia": 0.24, "Pneumothorax": 0.06, "Consolidation": 0.17,
        "Edema": 0.45, "Emphysema": 0.04, "Fibrosis": 0.08,
        "Pleural_Thickening": 0.12, "Hernia": 0.02,
    },
    "vit": {
        "Atelectasis": 0.31, "Cardiomegaly": 0.83, "Effusion": 0.48,
        "Infiltration": 0.22, "Mass": 0.11, "Nodule": 0.07,
        "Pneumonia": 0.21, "Pneumothorax": 0.04, "Consolidation": 0.14,
        "Edema": 0.40, "Emphysema": 0.03, "Fibrosis": 0.06,
        "Pleural_Thickening": 0.10, "Hernia": 0.01,
    },
}


def _placeholder_predict(model_key: str) -> Dict[str, float]:
    if model_key == "ensemble":
        base_probs = {d: 0.0 for d in _PLACEHOLDER_BASE["densenet"].keys()}
        for k in SUPPORTED_MODELS:
            for d, p in _PLACEHOLDER_BASE[k].items():
                base_probs[d] += p / len(SUPPORTED_MODELS)
        base = base_probs
    else:
        base = _PLACEHOLDER_BASE.get(model_key, _PLACEHOLDER_BASE["densenet"])
        
    return {
        d: round(min(1.0, max(0.0, p + random.uniform(-0.04, 0.04))), 4)
        for d, p in base.items()
    }


# ── AI 판독문 초안 생성 ─────────────────────────────────────────────────────

DISEASE_KR: Dict[str, str] = {
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


DISEASE_REPORT_HINTS_KR: Dict[str, str] = {
    "Atelectasis": "폐 용적 감소 또는 선상 음영 여부를 확인하십시오.",
    "Cardiomegaly": "심장 음영 확대와 촬영 자세/AP portable 여부를 함께 확인하십시오.",
    "Effusion": "늑골횡격막각 둔화 또는 흉막강 액체 저류 가능성을 확인하십시오.",
    "Infiltration": "국소 또는 미만성 폐 음영 증가 여부를 확인하십시오.",
    "Mass": "국소 종괴성 음영이 실제 병변인지 추가 영상 또는 과거 영상과 비교하십시오.",
    "Nodule": "작은 결절성 음영 여부와 과거 영상 대비 변화를 확인하십시오.",
    "Pneumonia": "감염성 침윤 또는 경화 가능성을 임상 증상과 함께 확인하십시오.",
    "Pneumothorax": "흉막선과 폐혈관 음영 소실 여부를 우선 확인하십시오.",
    "Consolidation": "폐포성 경화 음영 및 air bronchogram 가능성을 확인하십시오.",
    "Edema": "혈관 울혈, 간질성 음영 증가, 심부전 관련 소견을 함께 확인하십시오.",
    "Emphysema": "과팽창, 혈관 음영 감소 등 만성 폐쇄성 변화 가능성을 확인하십시오.",
    "Fibrosis": "망상 음영 또는 구조 왜곡 여부를 과거 영상과 비교하십시오.",
    "Pleural_Thickening": "흉막 비후 또는 흉막 병변 가능성을 확인하십시오.",
    "Hernia": "횡격막 주변 비정상 공기/연부조직 음영 여부를 확인하십시오.",
}


DISEASE_REPORT_HINTS_EN: Dict[str, str] = {
    "Atelectasis": "Check for volume loss or linear opacity.",
    "Cardiomegaly": "Correlate cardiac silhouette enlargement with projection and portable AP technique.",
    "Effusion": "Check for blunting of the costophrenic angle or pleural fluid.",
    "Infiltration": "Review for focal or diffuse increased pulmonary opacity.",
    "Mass": "Correlate a focal mass-like opacity with prior or additional imaging.",
    "Nodule": "Review for a small nodular opacity and interval change.",
    "Pneumonia": "Correlate possible infectious opacity with clinical symptoms.",
    "Pneumothorax": "Urgently check for a pleural line and absent peripheral vascular markings.",
    "Consolidation": "Review for air-space opacity or air bronchogram.",
    "Edema": "Check for vascular congestion or interstitial edema pattern.",
    "Emphysema": "Review for hyperinflation and reduced vascular markings.",
    "Fibrosis": "Compare reticular opacity or architectural distortion with prior imaging.",
    "Pleural_Thickening": "Review for pleural thickening or pleural-based abnormality.",
    "Hernia": "Check for abnormal diaphragmatic contour or adjacent soft-tissue/air density.",
}


def _pct(value: float) -> str:
    """확률 값을 판독문용 퍼센트 문자열로 변환합니다."""
    return f"{value * 100:.1f}%"


def _disease_kr(label: str) -> str:
    return DISEASE_KR.get(label, label.replace("_", " "))


def _build_case_id(contents: bytes) -> str:
    """환자 식별정보 없이 이미지 바이트 기반 케이스 ID를 생성합니다."""
    return f"CXR-{sha256(contents).hexdigest()[:12].upper()}"


def _build_report_draft(
    probs: Dict[str, float],
    detected: List[str],
    top_disease: str,
    threshold: float,
    is_placeholder: bool,
) -> Dict[str, object]:
    """
    확률 결과를 의료진이 복사·수정할 수 있는 판독문 초안으로 변환합니다.

    이 초안은 최종 진단이 아니라 판독 보조 텍스트입니다. 의도적으로
    단정 표현을 피하고, "가능성", "의심", "검토" 중심으로 작성합니다.
    """
    sorted_items = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    top_items = sorted_items[:3]
    top_prob = probs[top_disease]
    top_kr = _disease_kr(top_disease)
    top_findings = [f"{label.replace('_', ' ')} ({_pct(prob)})" for label, prob in top_items]
    top_findings_kr = [f"{_disease_kr(label)}({_pct(prob)})" for label, prob in top_items]

    if detected:
        detected_kr = ", ".join(f"{_disease_kr(label)}({_pct(probs[label])})" for label in detected)
        detected_en = ", ".join(f"{label.replace('_', ' ')} ({_pct(probs[label])})" for label in detected)
        findings_kr = (
            f"AI 분석에서 {top_kr} 가능성이 가장 높게 예측되었습니다({_pct(top_prob)}). "
            f"감지 임계값({_pct(threshold)}) 이상으로 표시된 소견은 {detected_kr}입니다. "
            f"{DISEASE_REPORT_HINTS_KR.get(top_disease, '원본 영상과 활성화 맵을 함께 확인하십시오')}"
        )
        impression_kr = (
            f"AI는 {top_kr}를 우선 검토 대상으로 제안합니다. "
            "동반 소견 가능성과 촬영 조건을 고려하여 최종 판독에서 확인하십시오."
        )
        findings_en = (
            f"AI analysis suggests {top_disease.replace('_', ' ')} as the highest-probability finding ({_pct(top_prob)}). "
            f"Findings above the detection threshold ({_pct(threshold)}) include {detected_en}. "
            f"{DISEASE_REPORT_HINTS_EN.get(top_disease, 'Review the original image and activation map together.')}"
        )
        impression_en = (
            f"AI suggests {top_disease.replace('_', ' ')} for priority clinician review. "
            "Please correlate with image quality, projection, prior imaging, and clinical context."
        )
    else:
        findings_kr = (
            f"감지 임계값({_pct(threshold)}) 이상으로 뚜렷하게 표시된 질환은 없습니다. "
            f"가장 높은 예측 항목은 {top_kr}({_pct(top_prob)})이며, 현재 설정에서는 낮은 우선순위 검토 대상으로 분류됩니다."
        )
        impression_kr = "AI 분석상 임계값 이상 주요 흉부 질환은 감지되지 않았습니다. 단, 최종 판독은 원본 영상과 임상 정보를 바탕으로 의료진이 확인해야 합니다."
        findings_en = (
            f"No finding exceeded the detection threshold ({_pct(threshold)}). "
            f"The highest-probability item was {top_disease.replace('_', ' ')} ({_pct(top_prob)}), classified as low-priority under the current threshold."
        )
        impression_en = "No major chest finding exceeded the AI detection threshold. Final interpretation should be confirmed by a clinician using the original image and clinical context."

    if is_placeholder:
        need_review_reason = "현재 결과는 Placeholder 데모 응답이므로 실제 임상 판단이나 재학습 데이터로 바로 사용하지 말고 UI/워크플로우 검증용으로만 활용하십시오."
    elif top_disease == "Pneumothorax" and top_prob >= threshold:
        need_review_reason = "기흉 가능성이 임계값 이상으로 표시되어 긴급 검토가 필요할 수 있습니다."
    elif top_prob >= 0.75:
        need_review_reason = "상위 예측 확률이 높아 해당 소견을 우선 확인하는 것이 좋습니다."
    elif len(detected) >= 3:
        need_review_reason = "여러 질환이 동시에 임계값 이상으로 표시되어 동반 소견 또는 모델 혼동 가능성을 함께 검토해야 합니다."
    elif detected:
        need_review_reason = "임계값 이상 소견이 있으나 확률만으로 확진할 수 없으므로 원본 영상, Grad-CAM, 임상 정보를 함께 확인해야 합니다."
    else:
        need_review_reason = "임계값 이상 질환은 없지만 촬영 품질, 과거 영상, 임상 증상에 따라 의료진 확인이 필요합니다."

    safety_note = "본 AI 결과는 최종 진단이 아니며, 의료진의 독립적인 영상 판독과 임상적 판단을 대체하지 않습니다."
    report_draft_kr = (
        "[AI 판독문 초안]\n"
        f"소견: {findings_kr}\n"
        f"결론: {impression_kr}\n"
        f"검토 필요 사유: {need_review_reason}\n"
        f"주의: {safety_note}"
    )
    report_draft_en = (
        "[AI-assisted draft report]\n"
        f"Findings: {findings_en}\n"
        f"Impression: {impression_en}\n"
        f"Review note: {need_review_reason}\n"
        f"Safety note: {safety_note}"
    )

    return {
        "Top_Findings": top_findings,
        "Top_Findings_KR": top_findings_kr,
        "Findings_KR": findings_kr,
        "Impression_KR": impression_kr,
        "Findings_EN": findings_en,
        "Impression_EN": impression_en,
        "Report_Draft_KR": report_draft_kr,
        "Report_Draft_EN": report_draft_en,
        "Need_Review_Reason": need_review_reason,
        "Safety_Note": safety_note,
    }


def _feedback_queue_size() -> int:
    if not FEEDBACK_QUEUE_PATH.exists():
        return 0
    with FEEDBACK_QUEUE_PATH.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# ── 실제 모델 추론 ────────────────────────────────────────────────────────────

def _real_predict(model_key: str, image: Image.Image) -> Dict[str, float]:
    tensor = preprocess_single_image(image).to(DEVICE)
    with torch.no_grad():
        if model_key == "ensemble":
            loaded_models = [m for m in _model_registry.values() if m is not None]
            if not loaded_models:
                raise ValueError("Ensemble 추론을 위한 모델이 로드되지 않았습니다.")
            probs_sum = torch.zeros(len(DISEASE_LABELS), device=DEVICE)
            for model in loaded_models:
                logits = model(tensor).squeeze(0)
                probs_sum += torch.sigmoid(logits)
            probs = (probs_sum / len(loaded_models)).cpu().tolist()
        else:
            model  = _model_registry[model_key]
            logits = model(tensor).squeeze(0)
            probs  = torch.sigmoid(logits).cpu().tolist()   # logits → 확률
            
    return {d: round(float(p), 4) for d, p in zip(DISEASE_LABELS, probs)}


# ── 엔드포인트 ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """서비스 상태 및 모델 로드 여부 반환."""
    loaded: List[str] = app.state.loaded_models
    return HealthResponse(
        status="healthy",
        model_loaded=len(loaded) > 0,
        model_version=f"v{API_VERSION}-{('ensemble' if len(loaded) > 1 else loaded[0]) if loaded else 'placeholder'}",
        loaded_models=loaded,
        version=API_VERSION,
    )


@app.get("/models", response_model=ModelInfoResponse, tags=["System"])
async def list_models():
    """지원 모델 목록 및 로드 상태 반환."""
    info = get_model_info()
    for key in info:
        if key == "ensemble":
            info[key]["is_loaded"] = any(m is not None for m in _model_registry.values())
        else:
            info[key]["is_loaded"] = _model_registry.get(key) is not None
    return ModelInfoResponse(models=info)


@app.get("/agent/status", tags=["Agentic Workflow"])
async def agent_runtime_status():
    """Expose non-secret LLM-agent runtime status for the Streamlit demo UI."""
    status = get_agent_runtime_status()
    status["loaded_cxr_models"] = [key for key, model in _model_registry.items() if model is not None]
    return status


def _parse_image_payload(contents: bytes, filename: str, content_type: Optional[str]) -> tuple[Image.Image, Dict[str, object], bool]:
    """Parse PNG/JPEG/DICOM bytes into PIL image plus safe metadata."""
    try:
        is_dicom_input = filename.lower().endswith((".dcm", ".dicom")) or is_dicom(io.BytesIO(contents))
        if is_dicom_input:
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            try:
                dicom_metadata = parse_dicom_metadata(tmp_path)
                image = dicom_to_pil(tmp_path)
            finally:
                os.unlink(tmp_path)
            return image, dicom_metadata, True

        allowed = ("image/png", "image/jpeg", "image/jpg", "application/octet-stream", None, "")
        if content_type and content_type not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"지원하지 않는 파일 형식: {content_type}. PNG/JPEG/DICOM을 사용하세요.",
            )
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return image, {}, False
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 처리 오류: {e}")


def _build_gradcam_overlay(model_key: str, image: Image.Image, top_disease: str, is_placeholder: bool) -> str:
    gradcam_b64 = _FAKE_GRADCAM_B64
    if is_placeholder:
        return gradcam_b64

    try:
        import numpy as np
        import cv2
        from src.analysis.gradcam import GradCAM, get_target_layer, apply_heatmap_overlay, cam_to_base64

        grad_model_key = model_key
        if model_key == "ensemble":
            loaded_keys = [k for k, v in _model_registry.items() if v is not None]
            if loaded_keys:
                grad_model_key = loaded_keys[0]

        target_model = _model_registry.get(grad_model_key)
        if target_model is not None:
            class_idx = DISEASE_LABELS.index(top_disease)
            tensor = preprocess_single_image(image).to(DEVICE)

            target_layer = get_target_layer(target_model)
            gcam = GradCAM(target_model, target_layer)
            cam = gcam.generate(tensor, class_idx, image_size=(image.height, image.width))
            gcam.remove_hooks()

            orig_img = np.array(image.convert("RGB"))
            if orig_img.shape[:2] != (image.height, image.width):
                orig_img = cv2.resize(orig_img, (image.width, image.height))
            overlay = apply_heatmap_overlay(orig_img, cam)
            gradcam_b64 = cam_to_base64(overlay)
    except Exception as e:
        print(f"Grad-CAM error: {e}")
    return gradcam_b64



def _build_case_agent_summary(
    *,
    filename: str,
    probs: Dict[str, float],
    detected: List[str],
    top_disease: str,
    triage: Dict[str, object],
    quality: Dict[str, object],
    is_placeholder: bool,
) -> str:
    """Compact single-case summary generated from dynamic tool observations."""
    top_prob = probs.get(top_disease, 0.0) * 100
    top_kr = _disease_kr(top_disease)
    detected_kr = ", ".join(_disease_kr(d) for d in detected) if detected else "없음"
    prefix = "[시연용] " if is_placeholder else ""
    triage_label = triage.get("triage_label_kr", "미분류") if isinstance(triage, dict) else "미분류"
    quality_grade = quality.get("quality_grade", "미실행") if isinstance(quality, dict) else "미실행"
    return (
        f"{prefix}{filename}: 동적 Agent가 선택 실행한 도구 기준 Top 소견은 "
        f"{top_kr}({top_prob:.1f}%)이며, 임계값 이상 소견은 {detected_kr}입니다. "
        f"Agent 판정은 {triage_label}, 영상 품질은 {quality_grade}입니다."
    )


def _build_dynamic_cxr_tools() -> List[CXRRuntimeTool]:
    """Create runtime tools that the dynamic agent can select per case."""

    def input_router(state: CXRCaseState) -> Dict[str, object]:
        image, dicom_metadata, is_dicom_input = _parse_image_payload(
            state.contents,
            state.filename,
            state.content_type,
        )
        return {
            "case_id": _build_case_id(state.contents),
            "image": image,
            "dicom_metadata": dicom_metadata,
            "is_dicom_input": is_dicom_input,
            "image_metadata": {
                "filename": state.filename,
                "content_type": state.content_type or "",
                "is_dicom_input": is_dicom_input,
                "width": image.width,
                "height": image.height,
                "dicom_metadata": dicom_metadata,
            },
        }

    def classifier_tool(state: CXRCaseState) -> Dict[str, object]:
        model_key = state.model_key
        if model_key not in API_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"지원하지 않는 모델: '{model_key}'. 지원 목록: {API_MODELS}",
            )
        image = state.get("image")
        start_ms = time.time()
        is_placeholder = False
        if model_key == "ensemble":
            if any(m is not None for m in _model_registry.values()):
                probs = _real_predict(model_key, image)
            else:
                is_placeholder = True
        else:
            if _model_registry.get(model_key) is not None:
                probs = _real_predict(model_key, image)
            else:
                is_placeholder = True

        if is_placeholder:
            time.sleep(0.3)
            probs = _placeholder_predict(model_key)

        inference_ms = int((time.time() - start_ms) * 1000)
        detected = [d for d, prob in probs.items() if prob >= state.threshold]
        top_disease = max(probs, key=probs.get)
        model_name = get_model_info().get(model_key, {}).get("display_name", model_key)
        return {
            "probs": probs,
            "detected": detected,
            "top_disease": top_disease,
            "top_probability": round(probs[top_disease], 4),
            "is_placeholder": is_placeholder,
            "inference_ms": inference_ms,
            "model_name": model_name,
        }

    def report_tool(state: CXRCaseState) -> Dict[str, object]:
        report = _build_report_draft(
            probs=dict(state.get("probs", {})),
            detected=list(state.get("detected", [])),
            top_disease=str(state.get("top_disease")),
            threshold=state.threshold,
            is_placeholder=bool(state.get("is_placeholder")),
        )
        return {"report": report}

    def gradcam_tool(state: CXRCaseState) -> Dict[str, object]:
        gradcam_b64 = _build_gradcam_overlay(
            state.model_key,
            state.get("image"),
            str(state.get("top_disease")),
            bool(state.get("is_placeholder")),
        )
        return {"gradcam_b64": gradcam_b64, "has_gradcam": not bool(state.get("is_placeholder"))}

    def quality_tool(state: CXRCaseState) -> Dict[str, object]:
        quality = analyze_image_quality(
            state.get("image"),
            is_dicom_input=bool(state.get("is_dicom_input")),
        )
        return {"quality_check": quality}

    def anatomy_tool(state: CXRCaseState) -> Dict[str, object]:
        anatomy = build_anatomy_assessment(
            state.get("image"),
            state.get("probs", {}),
            state.get("detected", []),
            str(state.get("top_disease")),
        )
        return {"anatomy_assessment": anatomy}

    def triage_tool(state: CXRCaseState) -> Dict[str, object]:
        triage = build_triage_assessment(
            state.get("probs", {}),
            state.get("detected", []),
            str(state.get("top_disease")),
            state.threshold,
            state.get("quality_check", {}),
            bool(state.get("is_placeholder")),
        )
        return {"triage_assessment": triage}

    return [
        CXRRuntimeTool(
            name="InputRouter",
            description="Route PNG/JPEG/DICOM payloads and extract safe image metadata.",
            requires=(),
            provides=("image", "dicom_metadata", "is_dicom_input", "case_id", "image_metadata"),
            run=input_router,
            required=True,
        ),
        CXRRuntimeTool(
            name="CXRClassifier",
            description="Run the selected CXR model or placeholder classifier and expose 14-label probabilities.",
            requires=("image",),
            provides=("probs", "detected", "top_disease", "top_probability", "is_placeholder", "inference_ms"),
            run=classifier_tool,
            required=True,
        ),
        CXRRuntimeTool(
            name="ReportDraftTool",
            description="Generate a clinician-editable Korean/English report draft from classifier observations.",
            requires=("probs", "detected", "top_disease", "is_placeholder"),
            provides=("report",),
            run=report_tool,
        ),
        CXRRuntimeTool(
            name="GradCAMTool",
            description="Generate Grad-CAM visual attribution when evidence/localization is useful.",
            requires=("image", "top_disease", "is_placeholder"),
            provides=("gradcam_b64", "has_gradcam"),
            run=gradcam_tool,
        ),
        CXRRuntimeTool(
            name="QualityCheckTool",
            description="Assess image size, brightness, contrast, entropy, sharpness and windowing limitations.",
            requires=("image", "is_dicom_input"),
            provides=("quality_check",),
            run=quality_tool,
        ),
        CXRRuntimeTool(
            name="AnatomicalROITool",
            description="Create a coarse anatomical review scaffold based on predicted findings.",
            requires=("image", "probs", "detected", "top_disease"),
            provides=("anatomy_assessment",),
            run=anatomy_tool,
        ),
        CXRRuntimeTool(
            name="TriageTool",
            description="Combine probabilities, critical findings, placeholder status and quality into review priority.",
            requires=("probs", "detected", "top_disease", "is_placeholder"),
            provides=("triage_assessment",),
            run=triage_tool,
        ),
    ]


def _run_prediction_pipeline(
    *,
    contents: bytes,
    filename: str,
    content_type: Optional[str],
    model_key: str,
    threshold: float,
    question: str = "",
) -> PredictionResult:
    """Shared dynamic single-image agent path used by /predict and /agent/analyze."""
    if model_key not in API_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 모델: '{model_key}'. 지원 목록: {API_MODELS}",
        )
    if not contents:
        raise HTTPException(status_code=400, detail="빈 파일이 업로드되었습니다.")

    state = CXRCaseState(
        contents=contents,
        filename=filename,
        content_type=content_type,
        model_key=model_key,
        threshold=threshold,
        question=question,
    )
    agent = DynamicCXRWorkflowAgent(_build_dynamic_cxr_tools())
    state = agent.run(state)
    data = state.data

    probs: Dict[str, float] = dict(data.get("probs") or {})
    detected: List[str] = list(data.get("detected") or [])
    top_disease = str(data.get("top_disease") or (max(probs, key=probs.get) if probs else ""))
    report: Dict[str, object] = dict(data.get("report") or {})
    quality_check: Dict[str, object] = dict(data.get("quality_check") or {})
    anatomy_assessment: Dict[str, object] = dict(data.get("anatomy_assessment") or {})
    triage_assessment: Dict[str, object] = dict(data.get("triage_assessment") or {})

    if not triage_assessment and probs:
        triage_assessment = build_triage_assessment(
            probs,
            detected,
            top_disease,
            threshold,
            quality_check,
            bool(data.get("is_placeholder")),
        )
    agent_summary = _build_case_agent_summary(
        filename=filename,
        probs=probs,
        detected=detected,
        top_disease=top_disease,
        triage=triage_assessment,
        quality=quality_check,
        is_placeholder=bool(data.get("is_placeholder")),
    ) if probs else "동적 Agent 실행 중 분류 결과를 생성하지 못했습니다."

    image_metadata = dict(data.get("image_metadata") or {})
    image_metadata["agent_plan"] = state.plan_history

    return PredictionResult(
        **probs,
        Case_ID=str(data.get("case_id")),
        Detected_Diseases=detected,
        Top_Disease=top_disease,
        Top_Probability=round(float(data.get("top_probability", probs.get(top_disease, 0.0))), 4),
        GradCAM_Base64=str(data.get("gradcam_b64") or _FAKE_GRADCAM_B64),
        Inference_Time_ms=int(data.get("inference_ms", 0)),
        Model_Used=str(data.get("model_name") or get_model_info().get(model_key, {}).get("display_name", model_key)),
        Model_Key=model_key,
        Is_Placeholder=bool(data.get("is_placeholder")),
        Report_Draft=str(report.get("Report_Draft_KR", "")),
        Findings_KR=str(report.get("Findings_KR", "")),
        Impression_KR=str(report.get("Impression_KR", "")),
        Need_Review_Reason=str(report.get("Need_Review_Reason", "")),
        Clinical_Report=report,
        Image_Metadata=image_metadata,
        Quality_Check=quality_check,
        Anatomy_Assessment=anatomy_assessment,
        Triage_Assessment=triage_assessment,
        Agent_Summary=agent_summary,
        Agent_Tool_Trace=state.tool_trace,
        Agent_Plan=state.plan_history,
    )


@app.post("/predict", response_model=PredictionResult, tags=["Inference"])
async def predict(
    file: UploadFile = File(..., description="흉부 X-ray (PNG/JPEG) 또는 DICOM (.dcm)"),
    model: str = Query(
        default="ensemble",
        description="사용할 모델: ensemble | densenet | efficientnet | vit",
    ),
    threshold: float = Query(
        default=DETECTION_THRESHOLD,
        ge=0.0, le=1.0,
        description="질환 감지 임계값 (기본 0.3)",
    ),
):
    """흉부 X-ray 단일 이미지를 분석하고 Agent 보조 메타데이터까지 반환합니다."""
    model_key = model.lower().strip()
    contents = await file.read()
    return _run_prediction_pipeline(
        contents=contents,
        filename=file.filename or "uploaded_image",
        content_type=file.content_type,
        model_key=model_key,
        threshold=threshold,
    )


@app.post("/agent/analyze", response_model=AgentBatchResponse, tags=["Agentic Workflow"])
async def analyze_with_agent(
    files: List[UploadFile] = File(..., description="1개 이상의 흉부 X-ray 또는 DICOM 파일"),
    model: str = Query(
        default="ensemble",
        description="사용할 모델: ensemble | densenet | efficientnet | vit",
    ),
    threshold: float = Query(
        default=DETECTION_THRESHOLD,
        ge=0.0, le=1.0,
        description="질환 감지 임계값",
    ),
    question: str = Query(
        default="",
        description="의료진이 Agent에게 묻는 케이스 질문 또는 비교 요청",
    ),
):
    """MedRAX-style multi-image case workbench endpoint.

    기존 학습 모델은 그대로 사용하고, 런타임에서 여러 장의 이미지별
    예측·판독 초안·품질 점검·해부학 ROI·triage·비교 요약을 묶어 반환합니다.
    """
    model_key = model.lower().strip()
    if not files:
        raise HTTPException(status_code=400, detail="1개 이상의 파일을 업로드하세요.")
    if len(files) > 12:
        raise HTTPException(status_code=400, detail="한 번에 최대 12개 파일까지만 분석할 수 있습니다.")

    cases: List[AgentCaseResult] = []
    summary_inputs: List[Dict[str, object]] = []
    for upload in files:
        contents = await upload.read()
        prediction = _run_prediction_pipeline(
            contents=contents,
            filename=upload.filename or "uploaded_image",
            content_type=upload.content_type,
            model_key=model_key,
            threshold=threshold,
            question=question,
        )
        probs = {label: getattr(prediction, label) for label in DISEASE_LABELS}
        agent_profile = {
            "quality_check": prediction.Quality_Check,
            "anatomy_assessment": prediction.Anatomy_Assessment,
            "triage_assessment": prediction.Triage_Assessment,
            "dicom_metadata": prediction.Image_Metadata.get("dicom_metadata", {}),
            "agent_summary": prediction.Agent_Summary,
            "agent_plan": prediction.Agent_Plan,
            "tool_trace": prediction.Agent_Tool_Trace,
        }
        case = AgentCaseResult(
            filename=upload.filename or "uploaded_image",
            case_id=prediction.Case_ID,
            prediction=prediction,
            probabilities=probs,
            top_disease=prediction.Top_Disease,
            top_probability=prediction.Top_Probability,
            detected_diseases=prediction.Detected_Diseases,
            report_draft=prediction.Report_Draft,
            agent_profile=agent_profile,
            agent_tool_trace=prediction.Agent_Tool_Trace,
            agent_plan=prediction.Agent_Plan,
            is_placeholder=prediction.Is_Placeholder,
        )
        cases.append(case)
        summary_inputs.append(
            {
                "filename": case.filename,
                "case_id": case.case_id,
                "probabilities": probs,
                "top_disease": case.top_disease,
                "top_probability": case.top_probability,
                "detected_diseases": case.detected_diseases,
                "is_placeholder": case.is_placeholder,
                "agent_profile": agent_profile,
            }
        )

    batch_agent = DynamicBatchWorkflowAgent()
    batch_result = batch_agent.run(
        cases=summary_inputs,
        question=question,
        build_summary=lambda case_rows, q: build_agent_batch_summary(case_rows, question=q),
    )
    agent_summary = dict(batch_result.get("summary") or {})
    case_traces: List[Dict[str, object]] = []
    case_plans: List[Dict[str, object]] = []
    for case in cases:
        for item in case.agent_tool_trace:
            trace_item = dict(item)
            trace_item.setdefault("scope", "case")
            trace_item.setdefault("case_id", case.case_id)
            trace_item.setdefault("filename", case.filename)
            case_traces.append(trace_item)
        for item in case.agent_plan:
            plan_item = dict(item)
            plan_item.setdefault("scope", "case")
            plan_item.setdefault("case_id", case.case_id)
            plan_item.setdefault("filename", case.filename)
            case_plans.append(plan_item)
    tool_trace = case_traces + list(batch_result.get("tool_trace") or [])
    agent_plan = case_plans + [{"scope": "batch", "planned_tools": batch_result.get("planned_tools", [])}]
    return AgentBatchResponse(
        status="completed",
        model_key=model_key,
        threshold=threshold,
        case_count=len(cases),
        cases=cases,
        agent_summary=agent_summary,
        tool_trace=tool_trace,
        agent_plan=agent_plan,
        safety_note="본 Agent 결과는 최종 진단이 아니며 의료진 검토가 필요합니다.",
    )


@app.post("/agent/chat", response_model=AgentChatResponse, tags=["Agentic Workflow"])
async def chat_with_agent(request: AgentChatRequest):
    """LLM-backed MedRAX-style follow-up chat over existing CXR-CAD outputs.

    이 엔드포인트는 모델을 다시 학습하거나 새 추론을 실행하지 않습니다.
    /agent/analyze가 만든 multi-image 결과를 MedRAX식 tool context로 압축한 뒤,
    OpenAI-compatible LLM을 agent brain으로 사용해 후속 질문에 답합니다.
    LLM 설정이 없으면 동일 컨텍스트 기반 deterministic fallback을 반환합니다.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="질문을 입력하세요.")
    reply = generate_llm_agent_reply(
        question=request.question,
        agent_result=request.result,
        history=[message.model_dump() for message in request.history],
    )
    return AgentChatResponse(**reply)


@app.post("/feedback", response_model=FeedbackResponse, tags=["Clinical Workflow"])
async def submit_feedback(feedback: FeedbackRequest):
    """
    의료진 피드백을 JSONL 검수 큐에 저장합니다.

    이 큐는 즉시 모델을 재학습하지 않고, 의료진 검수·라벨 정제·규제 검토를
    거친 뒤 학습 데이터 후보로 사용할 수 있는 데모용 운영 기록입니다.
    """
    FEEDBACK_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    submitted_at = datetime.now(timezone.utc).isoformat()
    queue_id = f"FB-{sha256(f'{feedback.case_id}-{submitted_at}-{feedback.feedback_type}'.encode('utf-8')).hexdigest()[:12].upper()}"
    item = feedback.model_dump()
    item.update(
        {
            "queue_id": queue_id,
            "submitted_at": submitted_at,
            "review_status": "queued_for_clinical_review",
            "retraining_candidate": feedback.feedback_type in {"AI 판단 불일치", "히트맵 위치 부정확", "질환 라벨 수정"},
            "regulatory_note": "피드백은 재학습 후보 큐에만 저장됩니다. 실제 모델 업데이트 전에는 별도 검수와 규제 검토가 필요합니다.",
        }
    )

    with FEEDBACK_QUEUE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return FeedbackResponse(
        status="saved",
        message="의료진 피드백이 검수 큐에 저장되었습니다.",
        queue_id=queue_id,
        queue_size=_feedback_queue_size(),
        saved_path=str(FEEDBACK_QUEUE_PATH),
    )


@app.get("/feedback/queue", response_model=FeedbackQueueResponse, tags=["Clinical Workflow"])
async def list_feedback_queue(limit: int = Query(default=20, ge=1, le=200)):
    """최근 의료진 피드백 큐 항목을 반환합니다."""
    if not FEEDBACK_QUEUE_PATH.exists():
        return FeedbackQueueResponse(total_count=0, items=[])

    items = []
    with FEEDBACK_QUEUE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return FeedbackQueueResponse(total_count=len(items), items=items[-limit:][::-1])
