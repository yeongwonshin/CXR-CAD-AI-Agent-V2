"""
CXR-CAD 모델 아키텍처 정의.

실제 가중치는 Kaggle/Colab에서 학습 후 .pth 파일로 저장합니다.
가중치 로드는 api/main.py (서버 시작 시)에서 수행합니다.

⚠️  Forward 출력 규약 (중요):
  - 학습 시 : logits 반환 (sigmoid 없음) → FocalLoss/BCEWithLogitsLoss 와 올바르게 연동
  - 추론 시 : torch.sigmoid(model(x)) 로 확률 변환
  - API     : api/main.py 의 _real_predict() 에서 sigmoid 적용
  - Ensemble: SoftVotingEnsemble 이 sigmoid 내적 적용 후 평균

지원 아키텍처:
  - DenseNet121CAD  : DenseNet-121  (Input: 3×224×224 → Output: 14 logits)
  - EfficientNetCAD : EfficientNet-B4 (Input: 3×224×224 → Output: 14 logits)
  - ViTCAD          : ViT-B/16      (Input: 3×224×224 → Output: 14 logits)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torchvision import models as tv_models

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False


# ── 상수 ─────────────────────────────────────────────────────────────────────

DISEASE_LABELS: List[str] = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

NUM_CLASSES: int = len(DISEASE_LABELS)

SUPPORTED_MODELS: List[str] = ["densenet", "efficientnet", "vit"]


# ── DenseNet-121 ──────────────────────────────────────────────────────────────

class DenseNet121CAD(nn.Module):
    """
    DenseNet-121 기반 Multi-label 흉부 X-ray 분류 모델.

    - Backbone: DenseNet-121 (ImageNet pretrained)
    - Head    : Dropout(0.5) → Linear(1024 → 14)
    - Input   : (B, 3, 224, 224)
    - Output  : (B, 14) logits  ← 추론 시 torch.sigmoid() 적용
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = False):
        super().__init__()
        weights = tv_models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.densenet121(weights=weights)

        self.features = backbone.features
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(backbone.classifier.in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x).relu_()
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        return self.classifier(feat)  # logits — sigmoid는 추론 시 별도 적용


# ── EfficientNet-B4 ───────────────────────────────────────────────────────────

class EfficientNetCAD(nn.Module):
    """
    EfficientNet-B4 기반 Multi-label 흉부 X-ray 분류 모델.

    - Backbone: EfficientNet-B4 (ImageNet pretrained)
    - Head    : Dropout(0.4) → Linear(1792 → 14)
    - Input   : (B, 3, 224, 224)
    - Output  : (B, 14) logits  ← 추론 시 torch.sigmoid() 적용
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = False):
        super().__init__()
        weights = tv_models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.efficientnet_b4(weights=weights)

        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(backbone.classifier[1].in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)  # logits — sigmoid는 추론 시 별도 적용


# ── Vision Transformer (ViT-B/16) ─────────────────────────────────────────────

class ViTCAD(nn.Module):
    """
    Vision Transformer ViT-B/16 기반 Multi-label 분류 모델.

    - Backbone: torchvision ViT-B/16 (ImageNet pretrained)
    - Input : (B, 3, 224, 224)
    - Output: (B, 14) logits  ← 추론 시 torch.sigmoid() 적용
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = False):
        super().__init__()

        # timm/torchvision 구조 차이로 인한 가중치 로드 오류를 방지하기 위해 torchvision으로 통일합니다.
        weights = tv_models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.vit_b_16(weights=weights)
        in_features = backbone.heads.head.in_features  # 768
        backbone.heads.head = nn.Identity()
        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=0.1),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))  # logits — sigmoid는 추론 시 별도 적용


# ── 팩토리 함수 ───────────────────────────────────────────────────────────────

def build_model(
    model_name: str, num_classes: int = NUM_CLASSES, pretrained: bool = False
) -> nn.Module:
    """
    모델 이름으로 CAD 모델 인스턴스 생성.

    Args:
        model_name: "densenet" | "efficientnet" | "vit"
        num_classes: 출력 클래스 수 (기본 14)
        pretrained: ImageNet 사전학습 가중치 사용 여부
    """
    name = model_name.lower().strip()
    if name == "densenet":
        return DenseNet121CAD(num_classes=num_classes, pretrained=pretrained)
    elif name in ("efficientnet", "efficientnet-b4"):
        return EfficientNetCAD(num_classes=num_classes, pretrained=pretrained)
    elif name in ("vit", "vit-b/16", "vit_b_16"):
        return ViTCAD(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(
        f"지원하지 않는 모델: '{model_name}'. 지원 목록: {SUPPORTED_MODELS}"
    )


def get_model_info() -> Dict[str, Dict]:
    """UI/API 표시용 모델 메타데이터."""
    return {
        "ensemble": {
            "display_name": "Ensemble (Recommended)",
            "description": "로드된 모든 단일 모델의 결과를 Soft Voting 방식으로 앙상블하여 최고의 정확도를 제공합니다.",
            "params": "Combined",
            "input_size": "224×224",
            "icon": "✨",
        },
        "densenet": {
            "display_name": "DenseNet-121",
            "description": "Dense connectivity로 gradient vanishing 완화. 파라미터 효율적.",
            "params": "~8M",
            "input_size": "224×224",
            "icon": "🔗",
        },
        "efficientnet": {
            "display_name": "EfficientNet-B4",
            "description": "Compound scaling으로 정확도/효율 균형 최적화.",
            "params": "~19M",
            "input_size": "224×224",
            "icon": "⚡",
        },
        "vit": {
            "display_name": "ViT-B/16",
            "description": "Self-attention으로 전역 문맥 포착. 대규모 데이터에서 강력.",
            "params": "~86M",
            "input_size": "224×224",
            "icon": "🧠",
        },
    }
