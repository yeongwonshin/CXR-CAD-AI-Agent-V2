"""
Pydantic Schemas for CXR-CAD API.

요청/응답 모델 정의. 모델 선택 필드 추가.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ── 지원 모델 타입 ────────────────────────────────────────────────────────────

ModelType = Literal["ensemble", "densenet", "efficientnet", "vit"]


# ── 응답 스키마 ───────────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    """
    /predict 엔드포인트 응답 스키마.

    14개 질환별 확률, 요약 정보, Grad-CAM, 사용된 모델 정보 포함.
    """

    # ── 케이스 추적 ───────────────────────────────────────────────────────
    Case_ID:           str        = Field(..., description="업로드 이미지 기반 케이스 추적 ID")

    # ── 질환별 확률 ─────────────────────────────────────────────────────────
    Atelectasis:       float = Field(..., ge=0.0, le=1.0)
    Cardiomegaly:      float = Field(..., ge=0.0, le=1.0)
    Effusion:          float = Field(..., ge=0.0, le=1.0)
    Infiltration:      float = Field(..., ge=0.0, le=1.0)
    Mass:              float = Field(..., ge=0.0, le=1.0)
    Nodule:            float = Field(..., ge=0.0, le=1.0)
    Pneumonia:         float = Field(..., ge=0.0, le=1.0)
    Pneumothorax:      float = Field(..., ge=0.0, le=1.0)
    Consolidation:     float = Field(..., ge=0.0, le=1.0)
    Edema:             float = Field(..., ge=0.0, le=1.0)
    Emphysema:         float = Field(..., ge=0.0, le=1.0)
    Fibrosis:          float = Field(..., ge=0.0, le=1.0)
    Pleural_Thickening: float = Field(..., ge=0.0, le=1.0)
    Hernia:            float = Field(..., ge=0.0, le=1.0)

    # ── 요약 ────────────────────────────────────────────────────────────────
    Detected_Diseases: List[str]  = Field(..., description="임계값 이상 질환 목록")
    Top_Disease:       str        = Field(..., description="가장 높은 확률의 질환")
    Top_Probability:   float      = Field(..., ge=0.0, le=1.0, description="Top1 질환의 예측 확률")
    GradCAM_Base64:    str        = Field(..., description="Grad-CAM 히트맵 (Base64 PNG)")
    Inference_Time_ms: int        = Field(..., ge=0, description="추론 시간 (ms)")

    # ── 모델 정보 ────────────────────────────────────────────────────────────
    Model_Used:        str        = Field(..., description="사용된 모델 이름 (예: DenseNet-121)")
    Model_Key:         str        = Field(..., description="모델 키 (ensemble / densenet / efficientnet / vit)")
    Is_Placeholder:    bool       = Field(..., description="True이면 실제 체크포인트 추론이 아니라 Placeholder 데모 응답")

    # ── AI 판독문 초안 ────────────────────────────────────────────────────
    Report_Draft:      str        = Field(..., description="의료진이 복사·수정할 수 있는 한국어 AI 판독문 초안")
    Findings_KR:       str        = Field(..., description="한국어 Findings 초안")
    Impression_KR:     str        = Field(..., description="한국어 Impression 초안")
    Need_Review_Reason: str       = Field(..., description="의료진 검토가 필요한 이유 또는 주의사항")
    Clinical_Report:   Dict[str, Any] = Field(..., description="한국어/영문 판독문 초안과 주요 소견 요약")

    # ── MedRAX-style runtime agent enrichments ───────────────────────────────
    Image_Metadata: Dict[str, Any] = Field(default_factory=dict, description="입력 이미지/DICOM 메타데이터와 크기 정보")
    Quality_Check: Dict[str, Any] = Field(default_factory=dict, description="해상도·밝기·대비·선예도 기반 입력 영상 품질 점검")
    Anatomy_Assessment: Dict[str, Any] = Field(default_factory=dict, description="질환별 해부학적 ROI 검토 스캐폴드")
    Triage_Assessment: Dict[str, Any] = Field(default_factory=dict, description="Agent 기반 우선 검토 등급과 사유")
    Agent_Summary: str = Field(default="", description="해당 이미지에 대한 Agent 요약")
    Agent_Tool_Trace: List[Dict[str, Any]] = Field(default_factory=list, description="동적 Agent가 실제 실행한 도구 호출 로그")
    Agent_Plan: List[Dict[str, Any]] = Field(default_factory=list, description="동적 Agent planner의 iteration별 계획 기록")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "Atelectasis": 0.32, "Cardiomegaly": 0.85, "Effusion": 0.50,
                "Infiltration": 0.18, "Mass": 0.12, "Nodule": 0.08,
                "Pneumonia": 0.22, "Pneumothorax": 0.05, "Consolidation": 0.15,
                "Edema": 0.42, "Emphysema": 0.03, "Fibrosis": 0.07,
                "Pleural_Thickening": 0.11, "Hernia": 0.02,
                "Detected_Diseases": ["Cardiomegaly", "Effusion", "Edema"],
                "Top_Disease": "Cardiomegaly",
                "Top_Probability": 0.85,
                "GradCAM_Base64": "iVBORw0KGgoAAAANSUhEUg...",
                "Inference_Time_ms": 312,
                "Model_Used": "DenseNet-121",
                "Model_Key": "densenet",
                "Is_Placeholder": True,
                "Case_ID": "CXR-0123456789AB",
                "Report_Draft": "[AI 판독문 초안]\n소견: ...",
                "Findings_KR": "AI 분석에서 심비대 가능성이 가장 높게 예측되었습니다.",
                "Impression_KR": "AI는 심비대를 우선 검토 대상으로 제안합니다.",
                "Need_Review_Reason": "Placeholder 데모 응답이므로 실제 임상 판단에 사용할 수 없습니다.",
                "Clinical_Report": {
                    "Top_Findings": ["Cardiomegaly (85.0%)", "Effusion (50.0%)"],
                    "Findings_KR": "AI 분석에서 심비대 가능성이 가장 높게 예측되었습니다.",
                    "Impression_KR": "AI는 심비대를 우선 검토 대상으로 제안합니다.",
                    "Findings_EN": "AI analysis suggests cardiomegaly as the highest-probability finding.",
                    "Impression_EN": "AI suggests cardiomegaly for clinician review.",
                    "Report_Draft_KR": "[AI 판독문 초안]\n소견: ...",
                    "Report_Draft_EN": "[AI-assisted draft report]\nFindings: ...",
                    "Need_Review_Reason": "Placeholder 데모 응답이므로 실제 임상 판단에 사용할 수 없습니다.",
                    "Safety_Note": "본 결과는 최종 진단이 아니며 의료진 검토가 필요합니다.",
                },
            }]
        }
    }


class AgentCaseResult(BaseModel):
    """MedRAX-style multi-image agent case result."""

    filename: str = Field(..., description="업로드 파일명")
    case_id: str = Field(..., description="케이스 ID")
    prediction: PredictionResult = Field(..., description="기존 /predict와 동일한 단일 이미지 추론 결과")
    probabilities: Dict[str, float] = Field(default_factory=dict, description="질환별 확률 요약")
    top_disease: str = Field(..., description="Top1 질환")
    top_probability: float = Field(..., ge=0.0, le=1.0, description="Top1 질환 확률")
    detected_diseases: List[str] = Field(default_factory=list, description="임계값 이상 질환")
    report_draft: str = Field(default="", description="해당 이미지별 판독문 초안")
    agent_profile: Dict[str, Any] = Field(default_factory=dict, description="품질·해부학 ROI·triage·DICOM 메타데이터")
    agent_tool_trace: List[Dict[str, Any]] = Field(default_factory=list, description="해당 케이스에서 실제 실행된 동적 Agent tool trace")
    agent_plan: List[Dict[str, Any]] = Field(default_factory=list, description="해당 케이스의 planner iteration 기록")
    is_placeholder: bool = Field(..., description="Placeholder 응답 여부")


class AgentBatchResponse(BaseModel):
    """POST /agent/analyze 응답 스키마."""

    status: str = Field(..., description="처리 상태")
    model_key: str = Field(..., description="사용 모델 키")
    threshold: float = Field(..., ge=0.0, le=1.0, description="감지 임계값")
    case_count: int = Field(..., ge=0, description="분석된 이미지 수")
    cases: List[AgentCaseResult] = Field(default_factory=list, description="이미지별 결과")
    agent_summary: Dict[str, Any] = Field(default_factory=dict, description="다중 이미지 비교·triage·사용자 질문 기반 요약")
    tool_trace: List[Dict[str, Any]] = Field(default_factory=list, description="Agent 도구 호출 감사 로그")
    agent_plan: List[Dict[str, Any]] = Field(default_factory=list, description="배치 및 케이스 단위 planner 기록")
    safety_note: str = Field(default="본 결과는 최종 진단이 아니며 의료진 검토가 필요합니다.", description="안전 고지")


class AgentChatMessage(BaseModel):
    """Agent 대화 이력 메시지."""

    role: Literal["user", "agent", "assistant"] = Field(..., description="메시지 역할")
    content: str = Field(..., description="메시지 내용")


class AgentChatRequest(BaseModel):
    """POST /agent/chat 요청 스키마 — LLM 기반 후속 질의."""

    question: str = Field(..., description="사용자가 Agent에게 입력한 후속 질문")
    result: Dict[str, Any] = Field(..., description="/agent/analyze로 생성된 케이스 묶음 결과")
    history: List[AgentChatMessage] = Field(default_factory=list, description="현재 세션의 대화 이력")


class AgentChatResponse(BaseModel):
    """POST /agent/chat 응답 스키마."""

    answer: str = Field(..., description="LLM 또는 fallback agent의 답변")
    engine: str = Field(..., description="llm_openai_compatible 또는 local_grounded_fallback")
    model: Optional[str] = Field(default=None, description="사용된 LLM 모델명")
    fallback: bool = Field(default=False, description="LLM 호출 실패/미설정으로 fallback을 사용했는지 여부")
    used_context_tools: List[str] = Field(default_factory=list, description="답변 생성에 투입된 MedRAX식 tool context 목록")
    generated_at: Optional[str] = Field(default=None, description="응답 생성 시각")
    usage: Dict[str, Any] = Field(default_factory=dict, description="LLM 토큰 사용량 등 provider 응답 메타데이터")
    safety_note: str = Field(default="본 결과는 최종 진단이 아니며 의료진 검토가 필요합니다.", description="안전 고지")
    tool_trace: List[Dict[str, Any]] = Field(default_factory=list, description="후속 질문 답변 시 선택된 context tool trace")
    agent_plan: List[Dict[str, Any]] = Field(default_factory=list, description="후속 질문 답변 planner 기록")
    error: Optional[str] = Field(default=None, description="fallback 전환 사유 또는 LLM 오류 메시지")


FeedbackType = Literal[
    "AI 판단 동의",
    "AI 판단 불일치",
    "히트맵 위치 부정확",
    "질환 라벨 수정",
    "판독의 코멘트",
]


class FeedbackRequest(BaseModel):
    """POST /feedback 요청 스키마 — 의료진 피드백 및 재학습 검수 큐 입력."""

    case_id: str = Field(..., description="/predict 응답의 Case_ID")
    feedback_type: FeedbackType = Field(..., description="의료진 피드백 유형")
    original_top_disease: Optional[str] = Field(default=None, description="AI가 Top으로 제시한 질환")
    corrected_labels: List[str] = Field(default_factory=list, description="의료진이 수정한 질환 라벨")
    comment: str = Field(default="", description="판독의/검수자 코멘트")
    reviewer_id: Optional[str] = Field(default=None, description="판독의 또는 검수자 식별자")
    model_key: Optional[ModelType] = Field(default=None, description="예측에 사용된 모델 키")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="예측 당시 감지 임계값")
    prediction_summary: Dict[str, Any] = Field(default_factory=dict, description="확률, 감지 질환, 판독문 초안 등 재검수용 요약")


class FeedbackResponse(BaseModel):
    """POST /feedback 응답 스키마."""

    status: str = Field(..., description="저장 상태")
    message: str = Field(..., description="사용자 표시용 메시지")
    queue_id: str = Field(..., description="피드백 큐 항목 ID")
    queue_size: int = Field(..., ge=0, description="현재 큐에 저장된 총 항목 수")
    saved_path: str = Field(..., description="피드백 JSONL 저장 경로")


class FeedbackQueueResponse(BaseModel):
    """GET /feedback/queue 응답 스키마."""

    total_count: int = Field(..., ge=0, description="전체 피드백 큐 항목 수")
    items: List[Dict[str, Any]] = Field(default_factory=list, description="최근 피드백 항목")


class HealthResponse(BaseModel):
    """GET /health 응답 스키마."""
    status:        str       = Field(..., description="서비스 상태 (healthy / degraded)")
    model_loaded:  bool      = Field(..., description="모델 로드 여부")
    model_version: str       = Field(..., description="API 버전 (예: v1.0.0-ensemble)")
    loaded_models: List[str] = Field(default_factory=list, description="로드된 모델 키 목록")
    version:       str       = Field(..., description="API 버전")


class ModelInfoResponse(BaseModel):
    """GET /models 응답 스키마 — 지원 모델 목록."""
    models: dict = Field(..., description="지원 모델 정보 딕셔너리")
