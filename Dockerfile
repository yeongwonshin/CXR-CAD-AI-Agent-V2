# ──────────────────────────────────────────────────
# CXR-CAD: CUDA + PyTorch GPU 개발/배포 환경
# Base: pytorch/pytorch 2.2.0 + CUDA 12.1 + cuDNN 8
# ──────────────────────────────────────────────────
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# 시스템 의존성 (OpenCV, libGL 포함)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1-mesa-glx \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 설치 (캐시 최적화를 위해 requirements 먼저)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스코드 복사
COPY . .

# Python 패스 설정
ENV PYTHONPATH=/app

# 기본 포트
EXPOSE 8000 8501

# 기본 CMD: FastAPI 서버 (docker-compose로 오버라이드 가능)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
