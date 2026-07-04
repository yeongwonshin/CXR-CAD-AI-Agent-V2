# CXR-CAD - AI-Agent

[whatch the demo video](https://drive.google.com/file/d/1bgSycOwmrIX82I-phvk34ngxv-HMjbI0/view?usp=drive_link)

> End-to-end multi-label classification CAD system for 14 thoracic diseases  
> DenseNet-121 / EfficientNet-B4 / ViT-B/16 / NIH ChestX-ray14 Dataset

---

## Architecture

```text
+---------------------------+     HTTP/JSON      +----------------------+
|         Streamlit         | <----------------> |       FastAPI        |
|         Dashboard         |   localhost:8000   |       Backend        |
|        (port 8501)        |                    |  GET  /health        |
|                           |  ?model=ensemble   |  GET  /models        |
| Model Selection:          |       densenet     |  POST /predict       |
| [x] Ensemble (Recommended)|       efficientnet +----------+-----------+
| [ ] DenseNet              |       vit                     | auto-detect .pth
| [ ] EfficientNet          |                    +----------v-----------+
| [ ] ViT                   |                    |  checkpoints/         |
+---------------------------+                    |  densenet_best.pth    |
                                                 |  efficientnet_best.pth|
                                                 |  vit_best.pth         |
                                                 |  Placeholder if absent|
                                                 +----------------------+
```

## Project Structure

```text
CXR-CAD/
|-- Dockerfile                          # CUDA 12.1 + PyTorch 2.2.0 GPU environment
|-- docker-compose.yml                  # API + Dashboard multi-container setup
|-- requirements.txt                    # Full dependency list
|-- .env                                # Local environment variables such as API keys and paths
|-- configs/
|   `-- config.yaml                     # Training hyperparameters for model, data, and training settings
|
|-- scripts/
|   |-- download_data.sh                # Automatic dataset download through Kaggle API for NIH and CheXpert
|   |-- run_optuna.py                   # Vast.ai-based Optuna hyperparameter optimization
|   `-- train.py                        # Standalone Python training script
|
|-- src/
|   |-- preprocess/
|   |   |-- data_loader.py              # NIH CSV parsing, Patient-ID split, and pos_weight calculation
|   |   |-- dataset.py                  # PyTorch Dataset class for NIH ChestX-ray14
|   |   |-- split.py                    # Patient-level data split based on GroupKFold
|   |   |-- transforms.py               # CLAHE, training, inference, and TTA transform pipelines
|   |   `-- dicom_utils.py              # pydicom metadata parsing and DICOM-to-PIL conversion
|   |
|   |-- train/
|   |   |-- models.py                   # DenseNet-121, EfficientNet-B4, and ViT-B/16 definitions
|   |   |-- focal_loss.py               # Focal Loss with gamma=0,1,2 and pos_weight
|   |   |-- ensemble.py                 # Soft Voting Ensemble with three models
|   |   `-- trainer.py                  # 5-Fold GroupKFold, EarlyStopping, and Cosine Annealing
|   |
|   `-- analysis/
|       |-- evaluation.py               # AUROC/AUPRC, F1, and Confusion Matrix
|       |-- calibration.py              # ECE and Temperature Scaling
|       |-- gradcam.py                  # Grad-CAM for all three models and lung-region leakage detection
|       |-- subgroup.py                 # Subgroup analysis by gender, age group, and view position PA/AP
|       `-- external_val.py             # CheXpert domain-shift validation
|
|-- api/
|   |-- main.py                         # /health, /models, /predict with DICOM support
|   `-- schemas.py                      # Pydantic schemas for request and response models
|
|-- dashboard/
|   `-- app.py                          # Streamlit Dashboard with model selection UI
|
|-- notebooks/
|   |-- 01_EDA.ipynb                    # Data exploration and class distribution
|   |-- 02_CLAHE_Analysis.ipynb         # Visualization of preprocessing effects
|   |-- 03_Focal_Loss_Experiment.ipynb  # Experiment with the gamma parameter
|   |-- 04_Training.ipynb               # Colab training notebook
|   |-- 05_Operating_Point.ipynb        # Youden's J threshold optimization
|   |-- 06_Calibration.ipynb            # Temperature Scaling and ECE measurement
|   |-- 07_Subgroup_Analysis.ipynb      # Fairness evaluation by gender, age group, and view position PA/AP
|   |-- 08_External_Validation.ipynb    # CheXpert external validation
|   `-- 09_Error_Analysis.ipynb         # FP/FN and Shortcut Learning analysis
|
|-- checkpoints/                        # Ignored by .gitignore; stores .pth and .csv files by model subdirectory
|   |-- densenet/                       # DenseNet model weights and analysis result CSV files
|   |-- efficientnet/                   # EfficientNet model weights and analysis result CSV files
|   `-- vit/                            # ViT model weights and analysis result CSV files
|
`-- tests/
    |-- conftest.py                     # pytest fixtures
    |-- test_api.py                     # API endpoint integration tests
    |-- test_encoding.py                # Image encoding and decoding tests
    `-- test_transforms.py              # Preprocessing transform pipeline tests
```

## Deliverables Guide

The following summarizes the model weights and main analysis outputs currently managed in the `checkpoints/` folder.

| Path | Description |
|---|---|
| `checkpoints/<model_key>/<model_key>_best.pth` | Best-performing trained model weights |
| `checkpoints/<model_key>/test_predictions.csv` | Test-set model predictions and ground-truth probability values |
| `checkpoints/<model_key>/op_analysis.csv` | Operating Point threshold optimization results |
| `checkpoints/<model_key>/*_subgroup.csv` | Subgroup performance results by gender, age group, and view position |
| `checkpoints/<model_key>/domain_shift.csv` | External CheXpert domain-shift evaluation |
| `checkpoints/<model_key>/shortcut_regions.csv` | Grad-CAM-based Shortcut Learning distribution analysis |
| `checkpoints/<model_key>/false_*.csv` | Major False Positive and False Negative error cases |

## Additional Trained Model Weight Files

The trained model weight files, or `.pth checkpoints`, are additionally provided through the Google Drive link below.

> https://drive.google.com/drive/folders/1plw8yetMsk07lFYJAjxJjMwjel4-d-IQ?usp=drive_link

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
uvicorn api.main:app --reload --port 8000
```

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Start the Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard: [http://localhost:8501](http://localhost:8501)

### 4. Run the Full Stack with Docker

GPU is required.

```bash
docker-compose up --build
```

### 5. Run Tests

```bash
pytest tests/ -v
```

### 6. Hybrid Workflow

The recommended workflow combines Vast.ai and Kaggle.

To maximize cost efficiency, run heavy training and HPO on Vast.ai, and run lightweight statistics, analysis, and visualization tasks on Kaggle. For detailed environment setup, see [`VASTAI_SETUP.md`](VASTAI_SETUP.md). For the full workflow, see [`TEAM_WORKFLOW.md`](TEAM_WORKFLOW.md).

1. **Prepare data and run optimization on Vast.ai**:
   ```bash
   # Automatically download datasets, including NIH and CheXpert
   bash scripts/download_data.sh
   # Run Optuna hyperparameter optimization in a tmux background session
   python scripts/run_optuna.py --n_trials 50
   ```
2. **Run main training on Vast.ai**: Use the discovered configuration to run either single-fold or full training through the script, for example `python scripts/train.py --fold 1`.
3. **Upload and perform in-depth validation on Kaggle**: Upload the extracted optimal `.pth` weights as a Kaggle Private Dataset, then validate and visualize results in the free T4 environment using notebooks from `05_Operating_Point.ipynb` through `09_Error_Analysis.ipynb`.

After training is complete, model weights are saved in the format `checkpoints/<model_key>/<model_key>_best.pth`.

> **Placeholder mode**: If no `.pth` file exists in `checkpoints/<model_key>/`, the system returns simulated predictions.  
> Once a checkpoint is placed there, the server automatically switches to real inference without requiring a restart.

---

## Checkpoint Save Format

Standard format compatible with the Kaggle notebook training code:

```python
torch.save({
    "epoch"               : epoch,
    "model_state_dict"    : model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "val_auroc"           : best_auroc,
}, "checkpoints/<model_key>/<model_key>_best.pth")
```

The API server supports all three formats: `model_state_dict`, `state_dict`, and a direct state_dict.

---

## Supported Models

| Model | Parameters | Key Features |
|------|---------|------|
| **DenseNet-121** | ~8M | Dense connectivity; lightweight and fast |
| **EfficientNet-B4** | ~19M | Compound scaling; balanced accuracy and efficiency |
| **ViT-B/16** | ~86M | Global context learning based on Self-Attention |
| **Soft Voting Ensemble** | N/A | Average probability from three models |

Select a model through the `?model=ensemble|densenet|efficientnet|vit` parameter when calling the API.  
In the dashboard, select a model through the sidebar checkboxes or radio buttons.

---

## Detected Diseases

14 classes are supported.

| # | Disease | # | Disease |
|---|---------|---|---------|
| 1 | Atelectasis | 8 | Pneumothorax |
| 2 | Cardiomegaly | 9 | Consolidation |
| 3 | Effusion | 10 | Edema |
| 4 | Infiltration | 11 | Emphysema |
| 5 | Mass | 12 | Fibrosis |
| 6 | Nodule | 13 | Pleural Thickening |
| 7 | Pneumonia | 14 | Hernia |

---

## Example Results and Evaluation Metrics

> The numbers below reflect actual DenseNet-121 training results and evaluation data from `checkpoints/densenet/`.

---

<details>
<summary>1. Class Distribution and pos_weight</summary>

### 1. Class Distribution and pos_weight

| Disease | Count | Prevalence | pos_weight |
|---------|-------|------------|------------|
| Infiltration | 19,894 | 17.7% | 4.65 |
| Effusion | 13,317 | 11.9% | 7.42 |
| Atelectasis | 11,559 | 10.3% | 8.71 |
| Nodule | 6,331 | 5.6% | 16.86 |
| Pneumothorax | 5,302 | 4.7% | 20.28 |
| Mass | 5,782 | 5.1% | 18.61 |
| Consolidation | 4,667 | 4.2% | 22.83 |
| Pleural_Thick. | 3,385 | 3.0% | 32.33 |
| Cardiomegaly | 2,776 | 2.5% | 39.01 |
| Emphysema | 2,516 | 2.2% | 44.46 |
| Edema | 2,303 | 2.1% | 46.63 |
| Fibrosis | 1,686 | 1.5% | 65.57 |
| Pneumonia | 1,431 | 1.2% | 82.31 |
| Hernia | 227 | 0.2% | 492.42 |
| No Finding | 60,361 | 53.8% | N/A |

</details>

---

<details>
<summary>2. Focal Loss Gamma Experiment Results</summary>

### 2. Focal Loss Gamma Experiment Results

| gamma | Mean AUROC | Mean AUPRC | Change in AUROC | Interpretation |
|-------|------------|------------|----------------|------|
| 0 | 0.8159 | 0.2347 | N/A | Same as BCE. Best overall performance. |
| 1 | 0.8110 | 0.2207 | -0.0049 | Performance decreased when easy examples were suppressed. |
| 2 | 0.8094 | 0.2259 | -0.0065 | Performance decreased when training focused more on hard examples. |
| 3 | 0.8035 | 0.2113 | -0.0124 | Excessive focus caused a substantial performance drop. |

[Selected optimal gamma] gamma=0

- Reason: Both AUROC and AUPRC reached their maximum values. With pos_weight already applied, the additional Focal Loss gamma weighting was found to harm training stability.

</details>

---

<details>
<summary>3. Example 5-Fold Cross Validation Results</summary>

### 3. Example 5-Fold Cross Validation Results

```text
[Training] Model: DenseNet-121 (ImageNet Pretrained)
[Training] Focal Loss (gamma=0.0), pos_weight applied
```

| Fold | Val AUROC | Val AUPRC |
|------|-----------|-----------|
| 1 | 0.8134 | 0.3523 |
| 2 | 0.8056 | 0.3412 |
| 3 | 0.8201 | 0.3634 |
| 4 | 0.8089 | 0.3489 |
| 5 | 0.8145 | 0.3567 |
| Mean | 0.8125 | 0.3525 |
| Std | +/-0.0051 | +/-0.0074 |

</details>

---

<details>
<summary>4. Ensemble and TTA Example Results</summary>

### 4. Ensemble and TTA Example Results

**[Model Comparison]**

| Model | Mean AUROC | Mean AUPRC |
|-------|------------|------------|
| DenseNet-121 (Single) | 0.8125 | 0.3525 |
| EfficientNet-B4 (Single) | 0.8198 | 0.3612 |
| Ensemble (Soft Voting) | 0.8312 | 0.3756 |

**[TTA Effect]**

| Setting | Mean AUROC | Change |
|---------|------------|--------|
| Without TTA | 0.8312 | N/A |
| With TTA (H-Flip) | 0.8345 | +0.0033 |

[Conclusion] The Ensemble + TTA combination improved performance by +0.022 compared with a single model.

</details>

---

<details>
<summary>5. Operating Point Analysis Example for Cardiomegaly</summary>

### 5. Operating Point Analysis Example for Cardiomegaly

```text
[Operating Point] Cardiomegaly Analysis
```

| Criterion | Threshold | Sensitivity | Specificity | PPV | NPV |
|------|-----------|-------------|-------------|-----|-----|
| Youden's J | 0.24 | 0.856 | 0.836 | 0.099 | 0.996 |
| Sensitivity 90% | 0.01 | 1.000 | 0.306 | 0.030 | 1.000 |
| Specificity 90% | 0.70 | 0.668 | 0.936 | 0.180 | 0.993 |

**[Rationale for Operating Point Selection]**

1. **Screening use case** for general health checkups
   - Recommendation: Use an operating point near 90% sensitivity. The threshold should be lowered based on the values above.
   - Reason: Minimizing false negatives, or missed patients, is the highest priority.
   - Trade-off: False positives increase, which can lead to additional testing costs.

2. **Diagnostic support use case** for detailed examination of suspected patients
   - Recommendation: Use the 90% specificity criterion with Threshold=0.70.
   - Reason: This minimizes unnecessary additional testing and patient anxiety.
   - Trade-off: Some positive cases may be missed.

</details>

---

<details>
<summary>6. Calibration Example Results</summary>

### 6. Calibration Example Results

| Metric | Before Scaling | After Temp Scaling |
|--------|---------------|-------------------|
| ECE | 0.0823 | 0.0456 |
| MCE | 0.1234 | 0.0678 |

[Conclusion] Temperature Scaling reduced ECE below 0.05.  
Temperature = 1.8 (learned)

</details>

---

<details>
<summary>7. Subgroup Analysis Example Results</summary>

### 7. Subgroup Analysis Example Results

**[Subgroup] Gender Analysis**

| Disease | Male AUROC | Female AUROC | Gap | Cause Analysis |
|---------|------------|--------------|-----|----------|
| Cardiomegaly | 0.934 | 0.911 | +2.3% | Higher male prevalence |
| Effusion | 0.889 | 0.904 | -1.5% | Reflected characteristics of female data |
| Hernia | 0.919 | 0.929 | -1.0% | Variation caused by limited sample size |

**[Subgroup] Age Group Analysis**

| Age Group | Mean AUROC | Cause Analysis |
|-----------|------------|----------|
| Under 40 | 0.8498 | Best performance |
| 40-60 | 0.8394 | Largest amount of training data |
| Over 60 | 0.8079 | Harder discrimination due to complex comorbidities |

**[Subgroup] View Position Analysis**

| View | Mean AUROC | Gap vs PA | Cause Analysis |
|------|------------|-----------|----------|
| PA | 0.9416 | N/A | Standard acquisition condition and high quality |
| AP | 0.9019 | -4.0% | Emergency or critically ill portable imaging and lower image quality |

> **4.0% performance gap between PA and AP**: AP images are often portable emergency images, so their image quality is generally lower.  
> Recommended response: Consider separate AP-specific augmentation or domain adaptation methods.

</details>

---

<details>
<summary>8. External Validation Example Results</summary>

### 8. External Validation Example Results

```text
[External Validation] CheXpert Test Set (5,000 images)
```

| Disease | NIH AUROC | CheXpert AUROC | Gap |
|---------|-----------|----------------|-----|
| Atelectasis | 0.8256 | 0.8126 | -1.3% |
| Cardiomegaly | 0.9242 | 0.7798 | -14.4% |
| Consolidation | 0.8268 | 0.8814 | +5.5% |
| Edema | 0.9236 | 0.8126 | -11.1% |
| Effusion | 0.8962 | 0.8784 | -1.8% |
| Pneumonia | 0.7714 | 0.7403 | -3.1% |
| Pneumothorax | 0.8993 | 0.8593 | -4.0% |
| **Mean (macro_avg)** | **0.8667** | **0.8235** | **-4.3%** |

**[Domain Shift Cause Analysis]**

| Factor | NIH | CheXpert |
|------|-----|---------|
| Imaging institution | More than 30 institutions | Single Stanford institution |
| Labeling method | Automatic NLP with label noise | Radiologist review and uncertainty labels |
| Patient population | Mainly outpatient patients | Includes inpatients and higher severity cases |

**[Recommended Responses]**

- Fine-tuning: Add training on a subset of CheXpert data.
- Domain Adaptation: Apply Adversarial Training.
- Ensemble: Combine models trained on NIH and CheXpert.

</details>

---

<details>
<summary>9. Error Case Analysis Example with Grad-CAM</summary>

### 9. Error Case Analysis Example with Grad-CAM

**[Error Analysis] False Positive Top 5**

| Case | Image ID | Prediction | GT | Probability | Grad-CAM Analysis | Cause |
|------|----------|------|----|------|--------------|------|
| FP-1 | 00023456_002.png | Pneumothorax | Normal | 0.78 | Highlighted below the right clavicle | Clavicle border mistaken for pneumothorax boundary |
| FP-2 | 00034567_001.png | Cardiomegaly | Normal | 0.65 | Highlighted the whole heart | Normal enlarged cardiac silhouette in an obese patient |
| FP-3 | 00045678_003.png | Effusion | Normal | 0.72 | Highlighted the left lower region | Breast shadow mistaken for pleural effusion |
| FP-4 | 00056789_001.png | Nodule | Normal | 0.58 | Highlighted a dot in the right upper region | Vascular cross-section mistaken for a nodule |
| FP-5 | 00067890_002.png | Mass | Normal | 0.61 | Highlighted the left middle region | Imaging artifact |

**[Error Analysis] False Negative Top 5**

| Case | Image ID | Prediction | GT | Probability | Grad-CAM Analysis | Cause |
|------|----------|------|----|------|--------------|------|
| FN-1 | 00078901_001.png | Normal | Nodule | 0.12 | Focused on the heart region | Missed a small 5 mm nodule |
| FN-2 | 00089012_002.png | Normal | Pneumonia | 0.23 | Diffuse activation | Failed to recognize a diffuse lesion pattern |
| FN-3 | 00090123_001.png | Normal | Effusion | 0.18 | Focused on the upper lung | Missed a small pleural effusion |
| FN-4 | 00101234_003.png | Normal | Atelectasis | 0.21 | Ignored the left lung | Focused only on the right lung |
| FN-5 | 00112345_001.png | Normal | Hernia | 0.08 | Focused only on the lung area | Ignored the diaphragm region |

**[Lung-Region Leakage Analysis]**

- Total analyzed cases: 100
- Activation inside the lung region: 72 cases (72%)
- Activation outside the lung region: 28 cases (28%)
  - Highlighted bones such as clavicles and ribs: 12 cases
  - Highlighted medical devices such as pacemakers: 8 cases
  - Highlighted text or markers: 5 cases
  - Highlighted background: 3 cases

**[Shortcut Learning Assessment]**

- Cases highlighting medical devices or text are suspected Shortcut Learning cases.
- Recommended improvements: Apply masking and attention mechanisms.

</details>

---

## Tech Stack

| Category | Technology |
|------|------|
| **ML Framework** | PyTorch 2.2 / torchvision / timm |
| **Models** | DenseNet-121 / EfficientNet-B4 / ViT-B/16 |
| **Optimization** | Optuna (HPO) |
| **Preprocessing** | OpenCV (CLAHE) / pydicom / albumentations |
| **Evaluation** | scikit-learn / scipy |
| **Backend** | FastAPI / Pydantic / Uvicorn |
| **Frontend** | Streamlit / Plotly |
| **Infrastructure** | Vast.ai / Kaggle API / Docker / CUDA 12.1 |
| **Datasets** | NIH ChestX-ray14 for training and internal validation / CheXpert for external validation |
| **Testing** | pytest |
| **Configuration** | YAML (configs/config.yaml) |
