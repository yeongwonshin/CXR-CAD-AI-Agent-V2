# Dashboard Decision Loop

## 1. Overview

The dashboard is organized around a practical review loop rather than a static model demo:

```text
Upload image
  -> choose inference mode and threshold
  -> inspect prediction and report draft
  -> review Grad-CAM and clinical context
  -> compare cases or ask agent follow-up questions
  -> check reliability readiness
  -> save clinician feedback for review
```

## 2. Main Prediction Dashboard

### Purpose

The main dashboard supports single-image review. It connects API status, model selection, upload, prediction, visualization, and feedback.

### Key decisions

| Question | Dashboard output |
| --- | --- |
| Is the backend live? | API health status and loaded model information |
| Which inference mode is being used? | Selected model key and placeholder/real checkpoint state |
| Which findings are above threshold? | Detected disease list and probability chart |
| What is the most likely finding? | Top disease and top probability |
| Is the result ready to discuss? | Report draft, findings, impression, and review reason |
| Does a clinician disagree? | Feedback form with corrected labels, comments, and edited report |

### Review flow

1. Confirm API status.
2. Select inference mode and detection threshold.
3. Upload PNG, JPEG, or DICOM input.
4. Run prediction.
5. Review probabilities, detected labels, Grad-CAM, and report draft.
6. Submit clinician feedback when needed.

## 3. Agent Workbench

### Purpose

Agent Workbench is designed for multi-image or multi-case review. It runs the same prediction pipeline but adds case-level tool planning and batch comparison.

### Key decisions

| Question | Agent Workbench output |
| --- | --- |
| Which case should be reviewed first? | Triage assessment and cross-case priority summary |
| Are multiple files showing similar findings? | Case comparison and probability matrix |
| Is one image low quality? | Quality-check indicators per case |
| Is the model focusing on plausible anatomy? | Anatomy assessment and Grad-CAM context |
| What should be included in the draft report? | Per-image findings and impression draft |
| What does the agent base its answer on? | Tool trace and planned tool list |

### Review flow

1. Upload one or more images.
2. Provide an optional question, such as “Which case is most urgent?”
3. Run agent analysis.
4. Review per-case predictions and batch summary.
5. Ask follow-up questions in the agent chat.
6. Submit case-level feedback.

## 4. Analysis Results

### Purpose

The Analysis Results page turns checkpoint artifacts into model behavior summaries.

### Main artifacts

| Artifact | Meaning |
| --- | --- |
| `test_predictions.csv` | Test-set probabilities and labels |
| `op_analysis.csv` | Threshold and operating-point analysis |
| `*_subgroup.csv` | Gender, age, and view-position subgroup results |
| `domain_shift.csv` | NIH versus external validation results |
| `shortcut_regions.csv` | Grad-CAM shortcut-region distribution |
| `false_positive.csv`, `false_negative.csv` | Major error cases |

### Key decisions

- Which threshold balances sensitivity and specificity?
- Which subgroup has the largest performance gap?
- Which disease drops most under external validation?
- Which error cases are most important to inspect?
- Is an LLM summary useful for explaining metric changes?

## 5. Reliability Readiness

### Purpose

Reliability Readiness summarizes whether the current model artifacts look acceptable for routine monitoring, limited review, or blocked deployment.

### Key decisions

| Dimension | What is checked |
| --- | --- |
| Calibration | ECE and operating-point quality |
| Domain robustness | subgroup AUROC gaps and external validation drop |
| Localization | shortcut ratio and ROI-related concern |
| Hidden strata | underperforming clusters or hidden subgroups |

### Status interpretation

- **PASS**: no active warning or critical issue.
- **WARNING**: one or more warning signals should be reviewed before broad use.
- **CRITICAL**: at least one critical issue, or too many warnings, blocks deployment until resolved.

## 6. Feedback Queue

### Purpose

The feedback queue captures clinician review signals without automatically retraining the model.

### Supported feedback types

- AI judgment agreement,
- AI judgment disagreement,
- inaccurate heatmap location,
- disease-label correction,
- clinician comment.

### Why this matters

Feedback becomes an auditable review artifact. It can later support dataset curation, label correction, error review, and controlled retraining after proper validation.

## 7. One-line summary

CXR-CAD AI Agent is organized so that every screen answers a review question: **what did the model see, why might it matter, can we trust it, and what should a clinician review next?**
