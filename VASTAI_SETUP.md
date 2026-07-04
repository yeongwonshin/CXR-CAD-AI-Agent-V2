# Vast.ai 학습 및 HPO 셋업 가이드

본 문서는 캐글(Kaggle) 12시간 제한 환경을 벗어나, **Vast.ai**의 고성능 GPU 상에서 CXR-CAD 모델 학습과 Optuna 하이퍼파라미터 튜닝을 백그라운드로 안정적으로 구동하기 위한 방법을 안내합니다.

## 1. 인스턴스 대여 및 시작

Vast.ai에서 인스턴스를 빌릴 때 **PyTorch 2.x 베이스 이미지**를 권장합니다.
- 예시 Docker Image: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel` 등
- 필요 GPU: 5-Fold 전체 학습이나 Optuna 병렬 학습 등을 고려해 `RTX 3090 (24GB)` 이상 권장

인스턴스 SSH 연결 후 필요한 의존성을 설치합니다:
```bash
git clone https://github.com/sogang-cxr-cad/CXR-CAD.git
cd CXR-CAD

# 필수 패키지 설치
pip install -r requirements.txt
pip install optuna
```

## 2. 데이터셋 자동 다운로드 (Kaggle API)

수십 기가바이트에 달하는 데이터를 직접 업로드하는 대신, Kaggle 공식 파이썬 API를 사용하여 서버 다운로드 속도를 활용해 수 분 안에 데이터셋 세팅을 마칠 수 있습니다. 이를 위해 자동화 스크립트를 준비해 두었습니다.

### A. Kaggle API 토큰 발급 및 세팅
로컬 PC에서 인터넷 브라우저를 열고 다음을 수행합니다:
1. Kaggle.com 로그인 후 우측 상단 프로필 ➡ **Settings (Account)** 클릭.
2. API 섹션에서 **'Create New Token'** 버튼 클릭. (`kaggle.json` 다운로드 됨)
3. 다운로드된 `kaggle.json` 파일을 메모장으로 열어 내용을 복사.
4. **Vast.ai 서버 터미널**에서 다음 명령어로 파일을 생성해 복사한 내용을 붙여넣습니다:
```bash
mkdir -p ~/.kaggle
nano ~/.kaggle/kaggle.json
# 복사 붙여넣기 후 `Ctrl+O` -> `Enter` -> `Ctrl+X` 로 저장하며 종료
chmod 600 ~/.kaggle/kaggle.json
```

### B. 다운로드 스크립트 실행
준비된 쉘 스크립트를 실행하여 전체 이미지와 `.csv` 라벨 데이터를 자동으로 다운받고 알맞은 경로(`data/nih/`)에 정리합니다:
```bash
bash scripts/download_data.sh
```

## 3. Tmux를 활용한 백그라운드 무중단 실행

Vast.ai 특성상, 로컬 PC를 끄거나 SSH 연결이 끊기면 터미널 작업이 모두 강제 종료됩니다. 따라서 `tmux` 세션 안에서 코드를 실행하는 것이 **필수적**입니다.

### Tmux 세션 생성 및 접근
```bash
# 새로운 tmux 세션 생성 (이름: HPO)
tmux new -s HPO
```

(이제 `tmux` 화면 내에서 스크립트를 실행합니다.)

### A. 단일 Fold 전체 학습 스크립트 기반 실행 (`train.py`)
기존 `04_Training.ipynb`의 역할을 스탠드얼론으로 수행합니다:
```bash
python scripts/train.py --config configs/config.yaml --model densenet --fold 1
```

### B. Optuna 하이퍼파라미터 튜닝 (`run_optuna.py`)
수십 개의 설정을 자동으로 번갈아 시도하며 최고 조합을 찾아냅니다:
```bash
python scripts/run_optuna.py --config configs/config.yaml --n_trials 50 --study_name cxr_cad_v1
```

### Tmux 백그라운드 전환 (Detach) 및 복귀 (Attach)
- 코드를 실행한 뒤, 컴퓨터 전원을 끄고 싶다면 **`Ctrl` + `B`를 누르고 손을 뗀 뒤 `D`**를 누릅니다. (Detach)
- SSH 연결을 끊어도 학습 스크립트는 Vast.ai 서버에서 주야장천 돌아갑니다.
- 다시 로컬 PC를 켜고 접속한 뒤 진행 상황을 보려면 다음 명령어로 복귀합니다:
```bash
tmux attach -t HPO
```

## 4. 학습 체크포인트 확보

Optuna 또는 기본 학습 스크립트 실행이 종료되면 `checkpoints/` 폴더 내부에 `.pth` 파일들이 정상적으로 생성되었는지 확인 후, 이를 다시 로컬 PC로 다운로드 받아 FastAPI 모델 런타임에 활용합니다.
