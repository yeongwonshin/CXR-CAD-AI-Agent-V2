# 🚀 CXR-CAD 프로젝트 팀별 업무 가이드

현재 CXR-CAD 뼈대 코드는 구성되어 있으나, **학습된 AI 모델(가중치 파일)이 아직 없는 상태**입니다.
이 가이드는 본격적인 학습 및 통합 테스트를 위해 각 팀(Data, AI, BE, FE)이 진행해야 할 필수 점검 및 개발 단계를 정의합니다.

---

## 💡 0. 필수 숙지 원칙: `config.yaml`과 노트북 실험의 관계
- **초기 설정 및 기본 뼈대**: 데이터셋 경로(`kaggle.nih_dir`), 에포크 수, 시드(Seed) 등 공통 환경 변수는 `configs/config.yaml`에 고정하여 모든 노트북이 베이스라인으로 공유합니다.
- **탐색적 실험 (노트북 활용)**: 반면 `02_CLAHE`, `03_Focal_Loss`, `05_Operating_Point` 등의 튜닝 노트북에서는 반복문 등을 통해 여러 하이퍼파라미터(`clip_limit`, `gamma`, `threshold`)를 직접 대입해 보며 자유롭게 실험을 수행합니다.
- **최종 결과 반영 (업데이트)**: 노트북 실험을 통해 **최적값을 분석해 내면, 그 결과(예: 최적 gamma=2)를 다시 공유 파일인 `config.yaml`에 업데이트**합니다. 이후 진행될 전장 학습(`04_Training`)과 서빙 엔진은 오직 `config.yaml`에 기록된 최종 확정값만을 사용합니다.

**📌 노트북에서의 `config.yaml` 사용 예시 (초기 세팅)**  
모든 캐글 노트북 파일의 첫 번째 필수 코드 셀에는 아래처럼 YAML 파일을 읽어와 변수에 할당하는 초기 세팅 코드가 있어야 합니다.
```python
import yaml

# config.yaml 불러오기
with open("configs/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 하드코딩 대신 공통 변수 가져오기 (Single Source of Truth)
NIH_DIR = config['kaggle']['nih_dir']
BATCH_SIZE = config['train']['batch_size']
FOCAL_GAMMA = config['train']['focal_gamma']

print(f"Dataset Path: {NIH_DIR}, Batch Size: {BATCH_SIZE}, Gamma: {FOCAL_GAMMA}")
```

---

## 1. 📊 Data 팀
**목표:** 딥러닝 학습에 필요한 안정적인 전처리 파이프라인 및 데이터셋 환경 구축.

- [x] **`src/preprocess/split.py` 점검**
  - GroupKFold를 활용한 **환자 단위(Patient-level) 분할** 로직 검증.
  - 학습(Train), 검증(Val), 테스트(Test) 세트 간 데이터 누수(Data Leakage) 완벽 차단.
- [x] **`src/preprocess/data_loader.py` 점검**
  - PyTorch Dataset/DataLoader 구성 및 Transform(Augmentation) 파이프라인 확인.
  - CLAHE(대조도 보정) 로직 정상 동작 여부.
- [x] **Kaggle 데이터 준비 (`KAGGLE_SETUP.md` 필독)**
  - NIH ChestX-ray14 크기 축소 버전(224x224) 및 CheXpert 데이터 Kaggle 연동 환경 구성.
  - CheXpert Uncertain(-1) 레이블의 `u_zeros` 처리 로직 점검.
- [x] **데이터 탐색 및 전처리 검증 (Phase 1. 데이터 탐색 및 튜닝)**
  - 캐글 환경 구축 후, 제일 먼저 `notebooks/01_EDA.ipynb`를 실행하여 데이터셋의 질환 분포 및 Data Leakage를 검토합니다.
  - `notebooks/02_CLAHE_Analysis.ipynb`를 통하여 X-ray 영상 대조도 개선(CLAHE) 파라미터 튜닝 후 결과물을 AI팀에 공유합니다.

---

## 2. 🧠 AI 팀
**목표:** 모델 아키텍처 확정, 5-Fold 학습 루프 실행 및 최종 가중치 추출.

- [x] **`src/train/models.py` & `src/train/focal_loss.py` 점검**
  - DenseNet, EfficientNet, ViT 클래스가 `logits` 형태로 값을 제대로 출력하는지 점검. (Sigmoid 제거 확인).
  - 클래스 불균형 해결을 위한 Focal Loss 파라미터(`pos_weight`, `gamma`) 확인.
- [x] **Vast.ai GPU 학습 환경 구성 (`VASTAI_SETUP.md` 필독)**
  - 리소스 제한과 12시간의 제약을 돌파하기 위해 학습 파이프라인을 파이썬 스크립트(`scripts/train.py`, `run_optuna.py`)로 분리.
  - `download_data.sh` 스크립트를 통한 데이터 자동 다운로드 및 세팅 테스트 완료.
- [ ] **본격적인 학습 진행 (Phase 2. HPO 및 메인 모델 학습)**
  - Vast.ai의 고성능 GPU(RTX 3090/4090) 인스턴스에서 `tmux`를 활용, `scripts/run_optuna.py`를 실행하여 며칠간 최적의 파라미터를 도출.
  - 탐색된 최적 파라미터로 `scripts/train.py`를 수행하여 5-Fold를 완주하고 최고 성능 모델 가중치를 획득.
- [ ] **⚠️ 학습 모델 및 분석 결과 저장 규칙! (핵심)**
  - 학습 완료 후 최종 가중치 파일(`.pth`), 평가 결괏값(`.csv`) 등 산출물은 반드시 리포지토리의 **`checkpoints/<model>/` 폴더 내부에 저장**해야 Backend에서 즉시 로드할 수 있습니다.
  - 예시 파일: `checkpoints/densenet/densenet_best.pth`, `checkpoints/densenet/test_predictions.csv`
- [ ] **하이브리드 분석 워크플로우 (Kaggle)**
  - Vast.ai에서 생성 및 다운로드된 `.pth` 최적 가중치 파일들을 캐글에 Private Dataset으로 업로드합니다.
  - 비용 절감을 위해 추론/분석 작업은 캐글의 무료 T4 환경에서 진행하는 것을 권장합니다.
- [ ] **External Validation (Phase 3. 평가 및 검증)**
  - 가중치 데이터셋을 마운트하고, `notebooks/08_External_Validation.ipynb`에서 도메인 시프트(Domain Shift) 파악.
- [ ] **주제별 심층 분석 및 최적화 (Phase 1 & Phase 3. 노트북 분석 진행)**
  - **[Phase 1]** `notebooks/03_Focal_Loss_Experiment.ipynb`: 기본 γ 파라미터 탐색 (현행 Optuna 로 대체 가능).
  - **[Phase 3]** `notebooks/05_Operating_Point.ipynb`: 목적(스크리닝 vs 확진 보조)에 따른 Cut-off 임계값 결정.
  - **[Phase 3]** `notebooks/06_Calibration.ipynb`: Temperature Scaling을 이용한 모델 Confidence 확률 보정 수치 탐색.
  - **[Phase 3]** `notebooks/07_Subgroup_Analysis.ipynb`: 성별, 연령대 별 편향성(Fairness) 유무 검토.
  - **[Phase 3]** `notebooks/09_Error_Analysis.ipynb`: FP/FN 오답을 추출하고 Grad-CAM으로 Shortcut Learning 원인 분석.

---

## 3. ⚙️ BE (Backend) 팀
**목표:** AI 모델 가중치를 안정적으로 서빙하고, UI(Frontend)에 빠르고 정확한 API 리스폰스 제공.

- [ ] **`api/main.py` 구조 및 로직 점검**
  - AI팀이 생산해 `checkpoints/<model>/`에 배치할 `.pth` 파일을 자동으로 스캔하여 메모리에 적재하는 로직 점검 (Placeholder 모드 핸들링 포함).
  - API 추론 시, AI 모델이 출력한 `logits` 값을 `torch.sigmoid()`로 역산하여 확률화하는 로직 검증.
- [ ] **성능 모니터링 기능 추가 (Optional)**
  - 단일 이미지 당 API Response Time(Inference Time) 점검 및 속도 개선(Batching, TorchScript 등).
- [ ] **Soft Voting Ensemble 점검**
  - 여러 모델 가중치(`DenseNet`, `ViT` 등) 감지 시 예측 결과를 결합하는 과정(`src/train/ensemble.py`) 점검 및 연동.

---

## 4. 🎨 FE (Frontend) 팀
**목표:** 사용자 친화적이고 직관적인 분석 대시보드 UI 연동 및 예외 처리.

- [ ] **메인 대시보드 (`dashboard/app.py`) 점검**
  - `page_link` 버튼이나 사이드바에서 다른 페이지로 이동하는 UI 디자인(UX/UI 렌더링) 무너짐 없는지 테스트.
  - API 서버(BE)와 연동하여 임계값(Threshold)에 따른 동적 차트 업데이트 및 상태 알림 표시.
- [ ] **분석 결과 시각화 모듈 (`dashboard/pages/analysis_results.py`)** 
  - 최근 신규 생성된 멀티페이지 파일로, AI 팀이 `checkpoints/<model>/`에 저장할 `.csv` 파일들이 UI 차트로 정상 변환되는지 테스트.
  - 임시 예시 데이터(Example Data) 표시 안내 문구와, 실제 `.csv` 발견 시 자동 렌더링으로 넘어가는 전환 기능 확인.

---

## 📅 전체 실행 로드맵 (순서 권장)

1. **Phase 1 (Data & AI):** Data팀의 로더 검증 이후 ➡️ AI팀과 함께 `KAGGLE_SETUP.md` 구축 환경에 탑승.
2. **Phase 2 (AI 학습):** 5-Fold 학습 루프로 밤샘 GPU 학습 후 최고 모델을 `checkpoints/<model>/` 에 추출.
3. **Phase 3 (BE 서빙):** 가중치 파일 확보 후 BE팀은 로컬 API 환경에서 로드 테스트 진행.
4. **Phase 4 (FE 연동):** BE 서버(8000 포트) 실행 상태에서 Streamlit(8501 포트) FE팀 테스트, End-to-End 데모 완성.
