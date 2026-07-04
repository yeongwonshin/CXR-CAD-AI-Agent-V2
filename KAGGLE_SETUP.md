# 🚀 Kaggle 학습 설정 가이드

NIH ChestX-ray14로 학습하고 CheXpert로 External Validation을 수행하는  
전체 Kaggle 설정 방법을 단계별로 설명합니다.

---

## Step 1. NIH ChestX-ray14 (224×224 리사이즈 버전) 데이터셋 연결

1. **Kaggle에서 데이터셋 검색**
   - [kaggle.com/datasets](https://www.kaggle.com/datasets) 접속
   - `nih chest xray 224` 또는 `nih chest xray resized` 검색
   - 찾은 데이터셋의 **slug** 확인  
     예: `username/nih-chest-xrays-224`

2. **Notebook에서 데이터셋 추가**
   - Kaggle Notebook 우측 패널 → **+ Add Data** 클릭
   - 검색 후 해당 데이터셋 **Add** 클릭
   - 마운트 경로 확인: `/kaggle/input/{slug}/`

3. **`configs/config.yaml` 수정**  
   ```yaml
   kaggle:
     nih_dir: "/kaggle/input/{실제-slug}"  # ← 여기 변경
   ```

4. **이미지 디렉토리 구조 확인** (노트북 Cell 1 실행 후 출력 확인)  
   ```
   /kaggle/input/{slug}/
   ├── images/           # ← 이 경우 IMG_DIR 자동 감지됨
   ├── Data_Entry_2017_v2020.csv
   └── ...
   ```

---

## Step 2. CheXpert Private Dataset 업로드

CheXpert는 공개 배포가 제한되어 있어, 직접 신청 후 Kaggle에 업로드해야 합니다.

### 2-1. Stanford 라이선스 신청
1. [stanfordmlgroup.github.io/competitions/chexpert](https://stanfordmlgroup.github.io/competitions/chexpert/) 접속
2. **CheXpert-v1.0-small** (약 11 GB, JPEG 224×224) 버전 신청
3. 이메일로 다운로드 링크 수신 (보통 1~2일 이내)

### 2-2. 로컬에 다운로드

```bash
# 다운로드 링크로 wget 사용
wget -O chexpert-small.zip "https://download.link/from/email"
unzip chexpert-small.zip
```

다운로드 후 폴더 구조:
```
CheXpert-v1.0-small/
├── train/
│   ├── patient00001/
│   │   └── study1/
│   │       └── view1_frontal.jpg
│   └── ...
├── valid/
│   └── patient00001/...
├── train.csv
└── valid.csv
```

### 2-3. Kaggle Private Dataset으로 업로드

```bash
# Kaggle CLI 설치 (없으면)
pip install kaggle

# API 토큰 설정: kaggle.com → Account → API → Create New Token
# ~/.kaggle/kaggle.json 에 저장됨

# Dataset 생성 및 업로드
kaggle datasets init -p /path/to/CheXpert-v1.0-small
# dataset-metadata.json 이 생성됨 — 이름/설명 수정 후:
kaggle datasets create -p /path/to/CheXpert-v1.0-small --dir-mode zip
```

> ⚠️ 업로드 중 라이선스 옵션에서 **"Private"** 선택 필수  
> 공개 배포는 Stanford 라이선스 위반입니다.

### 2-4. Notebook에 CheXpert Dataset 연결

1. Kaggle Notebook → **+ Add Data** → **Your Datasets** 탭
2. 방금 업로드한 CheXpert 데이터셋 선택
3. 마운트 경로 확인: `/kaggle/input/{your-slug}/CheXpert-v1.0-small/`

4. **`configs/config.yaml` 수정**
   ```yaml
   kaggle:
     chexpert_dir: "/kaggle/input/{your-slug}/CheXpert-v1.0-small"
   ```

---

## Step 3. Kaggle Notebook 생성 및 설정

1. **Notebook 생성**
   - [kaggle.com/code](https://www.kaggle.com/code) → **+ New Notebook**
   - Type: **Notebook** (`.ipynb`)

2. **GPU 활성화**
   - 우측 설정 → **Accelerator: GPU T4 x2** (또는 x1)

3. **인터넷 허용** (git clone 필요)
   - Settings → **Internet** → **On**

4. **데이터셋 연결** (Step 1, 2에서 추가한 것들)

5. **노트북 파일 임포트 (Notebooks 폴더 활용)**  
   GitHub 저장소의 `notebooks/` 폴더 안에는 탐색(EDA), 튜닝, 학습, 검증을 위한 01번부터 09번까지의 노트북이 준비되어 있습니다.  
   각 작업 단계에 맞춰 필요한 노트북 파일을 Kaggle로 가져와 실행할 수 있습니다.
   ```
   방법 1: File → Import Notebook → GitHub URL → 레포 URL 입력 (예: notebooks/04_Training.ipynb)
   방법 2: 로컬에 clone한 리포지토리에서 노트북 파일(.ipynb)을 드래그 앤 드롭으로 업로드
   ```

---

## Step 4. 학습 실행 순서

```text
[Phase 1. 데이터 탐색 및 튜닝]
notebooks/01_EDA.ipynb                     # 데이터 분포 및 누수 점검
notebooks/02_CLAHE_Analysis.ipynb          # 전처리 시인성 평가
notebooks/03_Focal_Loss_Experiment.ipynb   # 임밸런스 대비 γ 튜닝

[Phase 2. 메인 모델 학습]
notebooks/04_Training.ipynb                # 5-Fold 학습 진행 → checkpoints/*.pth 추출
     ↓
(checkpoints/*.pth 및 .csv를 Kaggle Output Dataset으로 저장)
     ↓
[Phase 3. 평가 및 검증]
notebooks/05_Operating_Point.ipynb         # 최적 Threshold 산출
notebooks/06_Calibration.ipynb             # 모델 신뢰도(ECE) 보정
notebooks/07_Subgroup_Analysis.ipynb       # 연령/성별 편향성 검증
notebooks/08_External_Validation.ipynb     # 외부 도메인(CheXpert) 성능 검증
notebooks/09_Error_Analysis.ipynb          # Grad-CAM 기반 FP/FN 오류 분석
```

### 4-1. 학습 노트북 (`04_Training.ipynb`)

| 셀 | 내용 |
|----|------|
| Cell 0 | 환경 감지, git clone, pip install |
| Cell 1 | 경로 자동 탐색 (NIH_DIR, CHECKPOINT_DIR) |
| Cell 2 | 임포트 |
| Cell 3 | NIH CSV 로드, Patient Split, Leakage 확인 |
| Cell 4 | Class Distribution, pos_weight 계산 |
| Cell 5 | train_one_epoch / evaluate 함수 정의 |
| Cell 6 | Focal Loss γ 실험 (선택) |
| Cell 7 | 5-Fold GroupKFold 학습 ← 핵심 |
| Cell 8 | Test Set 평가 |
| Cell 9 | Ensemble 평가 |
| Cell 10 | 결과 요약 |

**학습 시간 예상 (Kaggle T4 기준)**

| 모델 | Epoch당 | 50 Epochs |
|------|---------|-----------|
| DenseNet-121 | ~8분 | ~7시간 |
| EfficientNet-B4 | ~12분 | ~10시간 |
| ViT-B/16 | ~18분 | ~15시간 |

> 각 모델 학습 후 **Save & Run All** → Kaggle이 자동으로 Output 저장

### 4-2. 체크포인트를 다음 노트북으로 전달

Kaggle에서 노트북 Output을 Dataset으로 만들어 연결:
```
학습 노트북 실행 완료
  → Notebook Output: /kaggle/working/checkpoints/<model>/<model>_best.pth 등
  → 해당 Output을 Dataset으로 저장 (우측 Output 탭 → "New Dataset")
  → 08_External_Validation.ipynb 에서 해당 Dataset 연결
```

---

## Step 5. External Validation (`08_External_Validation.ipynb`)

1. Cell 1에서 `CHECKPOINT_DIR` 를 학습 Output Dataset 경로로 지정  
   ```python
   CHECKPOINT_DIR = Path('/kaggle/input/cxr-cad-checkpoints')  # ← slug 변경
   ```

2. 전체 실행 → NIH vs CheXpert AUROC 비교 표 + 시각화 확인

---

## 요약 체크리스트

- [ ] NIH 224×224 dataset slug 확인 후 `config.yaml` 업데이트
- [ ] Stanford CheXpert 라이선스 신청 및 이메일 수신
- [ ] CheXpert Kaggle Private Dataset 업로드 완료
- [ ] `config.yaml` `chexpert_dir` 업데이트
- [ ] Kaggle Notebook GPU + 인터넷 활성화
- [ ] 데이터 탐색 및 튜닝 (01~03 노트북 필요에 따라 실행)
- [ ] `04_Training.ipynb` 실행 대용량 학습 진행
- [ ] 체크포인트를 Output Dataset으로 묶어서 저장 (`checkpoints/<model>/`)
- [ ] 이후 심층 분석 및 검증 (05~09 노트북) 실행 시 앞서 저장한 Output Dataset 연동
