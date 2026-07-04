"""
pydicom 기반 DICOM 메타데이터 파싱 유틸리티.

NIH ChestX-ray14 데이터셋은 PNG 형식이지만, 실제 임상 환경에서는
DICOM 파일이 입력될 수 있으므로 해당 파싱 유틸리티를 제공합니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    _PYDICOM_AVAILABLE = True
except ImportError:
    _PYDICOM_AVAILABLE = False


# DICOM 태그 → Python dict 매핑 대상 필드
_TAGS_OF_INTEREST = {
    "PatientID":       "patient_id",
    "PatientAge":      "patient_age",
    "PatientSex":      "patient_sex",
    "ViewPosition":    "view_position",     # PA / AP / LL
    "Modality":        "modality",          # CR / DX
    "Rows":            "rows",
    "Columns":         "columns",
    "PixelSpacing":    "pixel_spacing",     # mm/pixel
    "BitsAllocated":   "bits_allocated",
    "StudyDate":       "study_date",
    "Manufacturer":    "manufacturer",
}


def _require_pydicom() -> None:
    if not _PYDICOM_AVAILABLE:
        raise ImportError(
            "pydicom이 설치되어 있지 않습니다. `pip install pydicom` 실행 후 재시도하세요."
        )


def parse_dicom_metadata(dicom_path: str) -> Dict[str, Any]:
    """
    DICOM 파일에서 주요 메타데이터를 추출합니다.

    Args:
        dicom_path: DICOM 파일 경로 (.dcm)

    Returns:
        dict: 메타데이터 딕셔너리
            {
                "patient_id": str,
                "patient_age": int | None,
                "patient_sex": str,          # "M" | "F" | "Unknown"
                "view_position": str,        # "PA" | "AP" | "Unknown"
                "modality": str,
                "rows": int,
                "columns": int,
                "pixel_spacing": list[float] | None,
                "bits_allocated": int,
                "study_date": str | None,
                "manufacturer": str | None,
            }
    """
    _require_pydicom()

    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
    meta: Dict[str, Any] = {}

    for dicom_tag, key in _TAGS_OF_INTEREST.items():
        val = getattr(ds, dicom_tag, None)
        if val is not None:
            # pydicom Sequence → Python type 변환
            if hasattr(val, "original_string"):
                val = str(val)
            elif hasattr(val, "__iter__") and not isinstance(val, str):
                val = list(val)
            meta[key] = val
        else:
            meta[key] = None

    # Patient Age 정규화 (NIH 형식: "058Y" → 58)
    age = meta.get("patient_age")
    if isinstance(age, str):
        try:
            meta["patient_age"] = int(age.replace("Y", "").replace("M", "").strip())
        except ValueError:
            meta["patient_age"] = None

    return meta


def dicom_to_pil(dicom_path: str):
    """
    DICOM 파일을 PIL.Image.Image (RGB) 로 변환합니다.

    VOI LUT / Window Center / Width 적용 후 8bit 정규화.

    Args:
        dicom_path: DICOM 파일 경로

    Returns:
        PIL.Image.Image in RGB mode
    """
    _require_pydicom()
    from PIL import Image

    ds = pydicom.dcmread(dicom_path)
    pixel_array = ds.pixel_array.astype(np.float32)

    # VOI LUT 적용 (DICOM 규격 Windowing)
    try:
        pixel_array = apply_voi_lut(pixel_array, ds).astype(np.float32)
    except Exception:
        pass

    # 포토메트릭 해석: MONOCHROME1은 반전 필요
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        pixel_array = pixel_array.max() - pixel_array

    # 0~255 정규화
    pmin, pmax = pixel_array.min(), pixel_array.max()
    if pmax > pmin:
        pixel_array = (pixel_array - pmin) / (pmax - pmin) * 255.0
    pixel_array = pixel_array.astype(np.uint8)

    # Grayscale → RGB
    image = Image.fromarray(pixel_array, mode="L").convert("RGB")
    return image


def is_dicom(file_path) -> bool:
    """
    파일이 DICOM 형식인지 확인합니다 (확장자 또는 매직 바이트 기준).

    Args:
        file_path: 파일 경로 (str 또는 Path) 또는 BytesIO 객체.
    """
    import io as _io

    # BytesIO 객체인 경우: 매직 바이트만 검사
    if isinstance(file_path, _io.IOBase):
        try:
            current_pos = file_path.tell()
            file_path.seek(128)
            result = file_path.read(4) == b"DICM"
            file_path.seek(current_pos)  # 읽기 위치 복원
            return result
        except Exception:
            return False

    # 문자열/Path 경로인 경우
    path = Path(file_path)
    if path.suffix.lower() in (".dcm", ".dicom"):
        return True

    # 매직 바이트 확인: DICOM 파일은 offset 128에 "DICM" 문자열을 가짐
    try:
        with open(file_path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except (OSError, IOError):
        return False
