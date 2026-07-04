"""
Grad-CAM 구현 모듈.

DenseNet-121, EfficientNet-B4, ViT-B/16 공용 인터페이스 제공.
폐 영역 이탈 케이스 분류 로직 포함.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    DenseNet, EfficientNet, ViT 세 모델 모두 호환하는 공용 인터페이스.

    Reference:
        Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks" (2017)

    Args:
        model: CAD 모델 인스턴스
        target_layer: Gradient hook을 등록할 레이어
            - DenseNet-121  : model.features.denseblock4
            - EfficientNet  : model.features[-1]
            - ViT           : model.backbone.blocks[-1] (패치 기반)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._hooks: List = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: int,
        image_size: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """
        특정 클래스에 대한 Grad-CAM 히트맵 생성.

        Args:
            input_tensor: (1, 3, H, W) 입력 텐서
            class_idx: 대상 클래스 인덱스 (0~13)
            image_size: 출력 히트맵 크기 (H, W)

        Returns:
            numpy array (H, W) ∈ [0, 1] — 정규화된 히트맵
        """
        self.model.eval()
        device = input_tensor.device

        # Forward
        output = self.model(input_tensor)
        target = output[0, class_idx]

        # Backward
        self.model.zero_grad()
        target.backward(retain_graph=False)

        if self._gradients is None or self._activations is None:
            warnings.warn("Grad-CAM: gradient 또는 activation을 캡처하지 못했습니다.")
            return np.zeros(image_size)

        gradients   = self._gradients   # (1, C, H, W) or (1, N, C) for ViT
        activations = self._activations

        # ViT의 경우 패치 기반 처리
        if len(gradients.shape) == 3:
            return self._compute_vit_cam(gradients, activations, image_size)

        # CNN 계열 (DenseNet, EfficientNet)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # Global Average Pooling
        cam = (weights * activations).sum(dim=1, keepdim=False)[0]
        cam = torch.relu(cam).cpu().numpy()

        # 정규화 및 리사이징
        cam = self._normalize_and_resize(cam, image_size)
        return cam

    def _compute_vit_cam(
        self,
        gradients: torch.Tensor,
        activations: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> np.ndarray:
        """ViT 패치 기반 Grad-CAM (Gradient Rollout 간소화)."""
        # (1, N+1, C) → CLS 토큰 제외 → (1, N, C)
        grad = gradients[0, 1:]  # (N, C)
        act  = activations[0, 1:]  # (N, C)

        #weights = grad.mean(dim=-1)  # (N,)
        #cam = (weights.unsqueeze(-1) * act).sum(dim=-1)  # (N,)
        #cam = torch.relu(cam).cpu().numpy()

        weights = grad.mean(dim=-1)          # (N,)
        cam = (weights.unsqueeze(-1) * act).sum(dim=-1)  # (N,)
        cam = cam.abs().cpu().numpy()
        
        # 패치를 2D grid로 복원 (14×14 for ViT-B/16 + 224×224 input)
        n_patches = cam.shape[0]
        grid_size = int(n_patches ** 0.5)
        cam = cam.reshape(grid_size, grid_size)
        return self._normalize_and_resize(cam, image_size)

    @staticmethod
    def _normalize_and_resize(cam: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        resized = cv2.resize(cam, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
        return resized.astype(np.float32)


def get_target_layer(model: nn.Module) -> nn.Module:
    """
    모델 타입에 따른 Grad-CAM 대상 레이어 자동 선택.

    Returns:
        대상 레이어 모듈
    """
    model_class = type(model).__name__

    if model_class == "DenseNet121CAD":
        return model.features.denseblock4

    elif model_class == "EfficientNetCAD":
        return model.features[-1]

    elif model_class == "ViTCAD":
        if hasattr(model.backbone, "blocks"):
            return model.backbone.blocks[-1]
        elif hasattr(model.backbone, "encoder"):
            return model.backbone.encoder.layers[-1]
        else:
            raise AttributeError("ViT 모델에서 적절한 레이어를 찾을 수 없습니다.")

    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_class}")


def apply_heatmap_overlay(
    original_image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    원본 이미지에 Grad-CAM 히트맵 오버레이.

    Args:
        original_image: (H, W, 3) uint8 RGB 이미지
        cam:            (H, W) ∈ [0, 1] Grad-CAM 히트맵
        alpha: 히트맵 투명도 (0=원본만, 1=히트맵만)

    Returns:
        (H, W, 3) uint8 오버레이 이미지
    """
    heatmap = (cam * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap_rgb + (1 - alpha) * original_image).astype(np.uint8)
    return overlay


def cam_to_base64(cam_image: np.ndarray) -> str:
    """
    Grad-CAM 오버레이 이미지를 Base64 문자열로 변환 (API 응답용).

    Args:
        cam_image: (H, W, 3) uint8 RGB numpy 이미지

    Returns:
        Base64 인코딩 PNG 문자열
    """
    import base64
    import io
    pil_img = Image.fromarray(cam_image)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ── 폐 영역 이탈 케이스 분석 ──────────────────────────────────────────────────

LUNG_REGION_THRESHOLD = 0.4  # 폐 영역 활성화 비율 임계값


def detect_lung_deviation(
    cam: np.ndarray,
    lung_mask: Optional[np.ndarray] = None,
    image_size: Tuple[int, int] = (224, 224),
    center_fraction: float = 0.6,
) -> Dict:
    """
    Grad-CAM 활성화가 폐 영역을 벗어난 케이스 감지.

    폐 마스크가 없을 경우, 이미지 중앙 영역을 폐 영역으로 근사.

    Args:
        cam:              Grad-CAM 히트맵 (H, W) ∈ [0, 1]
        lung_mask:        폐 영역 이진 마스크 (H, W), None이면 중앙 근사
        center_fraction:  중앙 영역 비율 (lung_mask=None인 경우)

    Returns:
        {
            "lung_activation_ratio": float,     # 폐 영역 내 활성화 비율
            "is_deviated": bool,                # 폐 영역 이탈 여부
            "peak_location": tuple(y, x),       # 최대 활성화 좌표
        }
    """
    H, W = cam.shape

    if lung_mask is None:
        # 중앙 영역을 폐로 근사
        margin_h = int(H * (1 - center_fraction) / 2)
        margin_w = int(W * (1 - center_fraction) / 2)
        lung_mask = np.zeros((H, W), dtype=bool)
        lung_mask[margin_h:H-margin_h, margin_w:W-margin_w] = True

    # 폐 영역 내 활성화 비율
    total_activation   = cam.sum()
    lung_activation    = (cam * lung_mask.astype(float)).sum()
    lung_ratio         = float(lung_activation / max(total_activation, 1e-8))

    # 최대 활성화 위치
    peak_y, peak_x = np.unravel_index(cam.argmax(), cam.shape)
    is_peak_in_lung = bool(lung_mask[peak_y, peak_x])

    return {
        "lung_activation_ratio": lung_ratio,
        "is_deviated":           lung_ratio < LUNG_REGION_THRESHOLD or not is_peak_in_lung,
        "peak_location":         (int(peak_y), int(peak_x)),
        "peak_in_lung":          is_peak_in_lung,
    }
