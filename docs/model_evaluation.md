# Model Evaluation

## 1. Supported disease labels

The project supports 14 thoracic findings.

| # | Disease | # | Disease |
| --- | --- | --- | --- |
| 1 | Atelectasis | 8 | Pneumothorax |
| 2 | Cardiomegaly | 9 | Consolidation |
| 3 | Effusion | 10 | Edema |
| 4 | Infiltration | 11 | Emphysema |
| 5 | Mass | 12 | Fibrosis |
| 6 | Nodule | 13 | Pleural Thickening |
| 7 | Pneumonia | 14 | Hernia |

## 2. Deliverable artifacts

| Path | Description |
| --- | --- |
| `checkpoints/<model_key>/<model_key>_best.pth` | Best-performing trained model weights |
| `checkpoints/<model_key>/test_predictions.csv` | Test-set predictions and ground-truth values |
| `checkpoints/<model_key>/op_analysis.csv` | Operating-point threshold optimization results |
| `checkpoints/<model_key>/*_subgroup.csv` | Subgroup performance results by gender, age group, and view position |
| `checkpoints/<model_key>/domain_shift.csv` | External CheXpert domain-shift evaluation |
| `checkpoints/<model_key>/shortcut_regions.csv` | Grad-CAM-based shortcut-learning distribution analysis |
| `checkpoints/<model_key>/false_*.csv` | Major false-positive and false-negative error cases |

## 3. Class distribution and positive weight example

| Disease | Count | Prevalence | pos_weight |
| --- | ---: | ---: | ---: |
| Infiltration | 19,894 | 17.7% | 4.65 |
| Effusion | 13,317 | 11.9% | 7.42 |
| Atelectasis | 11,559 | 10.3% | 8.71 |
| Nodule | 6,331 | 5.6% | 16.86 |
| Pneumothorax | 5,302 | 4.7% | 20.28 |
| Mass | 5,782 | 5.1% | 18.61 |
| Consolidation | 4,667 | 4.2% | 22.83 |
| Pleural_Thickening | 3,385 | 3.0% | 32.33 |
| Cardiomegaly | 2,776 | 2.5% | 39.01 |
| Emphysema | 2,516 | 2.2% | 44.46 |
| Edema | 2,303 | 2.1% | 46.63 |
| Fibrosis | 1,686 | 1.5% | 65.57 |
| Pneumonia | 1,431 | 1.2% | 82.31 |
| Hernia | 227 | 0.2% | 492.42 |
| No Finding | 60,361 | 53.8% | N/A |

## 4. Focal Loss gamma experiment example

| gamma | Mean AUROC | Mean AUPRC | Change in AUROC | Interpretation |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0.8159 | 0.2347 | N/A | Same as BCE; best overall performance |
| 1 | 0.8110 | 0.2207 | -0.0049 | Performance decreased when easy examples were suppressed |
| 2 | 0.8094 | 0.2259 | -0.0065 | Performance decreased when training focused more on hard examples |
| 3 | 0.8035 | 0.2113 | -0.0124 | Excessive focus caused a substantial performance drop |

Selected gamma: `0`.

Interpretation: with `pos_weight` already applied, additional focal weighting reduced training stability in the recorded experiment.

## 5. Example cross-validation result

```text
Training model: DenseNet-121 with ImageNet pretrained weights
Loss: Focal Loss with gamma=0.0 and pos_weight
```

| Fold | Val AUROC | Val AUPRC |
| ---: | ---: | ---: |
| 1 | 0.8134 | 0.3523 |
| 2 | 0.8056 | 0.3412 |
| 3 | 0.8201 | 0.3634 |
| 4 | 0.8089 | 0.3489 |
| 5 | 0.8145 | 0.3567 |
| Mean | 0.8125 | 0.3525 |
| Std | ±0.0051 | ±0.0074 |

## 6. Ensemble and TTA example

| Model | Mean AUROC | Mean AUPRC |
| --- | ---: | ---: |
| DenseNet-121 single model | 0.8125 | 0.3525 |
| EfficientNet-B4 single model | 0.8198 | 0.3612 |
| Soft-voting ensemble | 0.8312 | 0.3756 |

| Setting | Mean AUROC | Change |
| --- | ---: | ---: |
| Without TTA | 0.8312 | N/A |
| With horizontal-flip TTA | 0.8345 | +0.0033 |

Interpretation: the recorded ensemble plus TTA example improved mean AUROC compared with a single baseline model.

## 7. Operating-point example for Cardiomegaly

| Criterion | Threshold | Sensitivity | Specificity | PPV | NPV |
| --- | ---: | ---: | ---: | ---: | ---: |
| Youden’s J | 0.24 | 0.856 | 0.836 | 0.099 | 0.996 |
| Sensitivity 90% | 0.01 | 1.000 | 0.306 | 0.030 | 1.000 |
| Specificity 90% | 0.70 | 0.668 | 0.936 | 0.180 | 0.993 |

### Interpretation

- Screening workflow: prefer a high-sensitivity operating point to reduce missed positive cases.
- Diagnostic-support workflow: prefer a high-specificity operating point to reduce unnecessary additional testing.

## 8. Calibration example

| Metric | Before scaling | After temperature scaling |
| --- | ---: | ---: |
| ECE | 0.0823 | 0.0456 |
| MCE | 0.1234 | 0.0678 |

Recorded temperature value: `1.8`.

Interpretation: temperature scaling reduced ECE below the 0.05 readiness warning threshold in this example.

## 9. Subgroup analysis example

### Gender

| Disease | Male AUROC | Female AUROC | Gap | Cause analysis |
| --- | ---: | ---: | ---: | --- |
| Cardiomegaly | 0.934 | 0.911 | +2.3% | Higher male prevalence |
| Effusion | 0.889 | 0.904 | -1.5% | Reflected characteristics of female data |
| Hernia | 0.919 | 0.929 | -1.0% | Variation caused by limited sample size |

### Age group

| Age group | Mean AUROC | Cause analysis |
| --- | ---: | --- |
| Under 40 | 0.8498 | Best performance |
| 40–60 | 0.8394 | Largest amount of training data |
| Over 60 | 0.8079 | Harder discrimination due to complex comorbidities |

### View position

| View | Mean AUROC | Gap vs PA | Cause analysis |
| --- | ---: | ---: | --- |
| PA | 0.9416 | N/A | Standard acquisition condition and high quality |
| AP | 0.9019 | -4.0% | Portable emergency imaging and lower image quality |

Recommended response: consider AP-specific augmentation, domain adaptation, or separate monitoring when the PA/AP gap is large.

## 10. External validation example

```text
External validation: CheXpert test set, 5,000 images
```

| Disease | NIH AUROC | CheXpert AUROC | Gap |
| --- | ---: | ---: | ---: |
| Atelectasis | 0.8256 | 0.8126 | -1.3% |
| Cardiomegaly | 0.9242 | 0.7798 | -14.4% |
| Consolidation | 0.8268 | 0.8814 | +5.5% |
| Edema | 0.9236 | 0.8126 | -11.1% |
| Effusion | 0.8962 | 0.8784 | -1.8% |
| Pneumonia | 0.7714 | 0.7403 | -3.1% |
| Pneumothorax | 0.8993 | 0.8593 | -4.0% |
| Mean macro average | 0.8667 | 0.8235 | -4.3% |

### Domain-shift factors

| Factor | NIH | CheXpert |
| --- | --- | --- |
| Imaging institution | More than 30 institutions | Single Stanford institution |
| Labeling method | Automatic NLP with label noise | Radiologist review and uncertainty labels |
| Patient population | Mainly outpatient patients | Includes inpatients and higher-severity cases |

Recommended responses include external-site fine-tuning, domain adaptation, and ensembles trained across datasets.

## 11. Error case and Grad-CAM review example

### False-positive examples

| Case | Image ID | Prediction | Ground truth | Probability | Grad-CAM observation | Possible cause |
| --- | --- | --- | --- | ---: | --- | --- |
| FP-1 | 00023456_002.png | Pneumothorax | Normal | 0.78 | Highlighted below the right clavicle | Clavicle border mistaken for pneumothorax boundary |
| FP-2 | 00034567_001.png | Cardiomegaly | Normal | 0.65 | Highlighted the whole heart | Normal enlarged cardiac silhouette in an obese patient |
| FP-3 | 00045678_003.png | Effusion | Normal | 0.72 | Highlighted the left lower region | Breast shadow mistaken for pleural effusion |
| FP-4 | 00056789_001.png | Nodule | Normal | 0.58 | Highlighted a dot in the right upper region | Vascular cross-section mistaken for a nodule |
| FP-5 | 00067890_002.png | Mass | Normal | 0.61 | Highlighted the left middle region | Imaging artifact |

### False-negative examples

| Case | Image ID | Prediction | Ground truth | Probability | Grad-CAM observation | Possible cause |
| --- | --- | --- | --- | ---: | --- | --- |
| FN-1 | 00078901_001.png | Normal | Nodule | 0.12 | Focused on the heart region | Missed a small 5 mm nodule |
| FN-2 | 00089012_002.png | Normal | Pneumonia | 0.23 | Diffuse activation | Failed to recognize a diffuse lesion pattern |
| FN-3 | 00090123_001.png | Normal | Effusion | 0.18 | Focused on the upper lung | Missed a small pleural effusion |
| FN-4 | 00101234_003.png | Normal | Atelectasis | 0.21 | Ignored the left lung | Focused only on the right lung |
| FN-5 | 00112345_001.png | Normal | Hernia | 0.08 | Focused only on the lung area | Ignored the diaphragm region |

## 12. Shortcut-learning summary example

- Total analyzed cases: 100
- Activation inside the lung region: 72 cases
- Activation outside the lung region: 28 cases
  - bones such as clavicles or ribs: 12 cases
  - medical devices such as pacemakers: 8 cases
  - text or markers: 5 cases
  - background: 3 cases

Recommended improvements include shortcut-prone sample cleaning, ROI-aware validation, masking, and attention-consistency review.
