# Decision Logic

## 1. Design philosophy

The core logic is designed around clinical review support, not autonomous diagnosis. The model output is treated as one input to a broader workflow that includes quality review, explanation context, thresholding, reliability assessment, and clinician feedback.

```text
Model probability
  + threshold policy
  + image quality context
  + metadata context
  + explanation context
  + reliability checks
  + clinician feedback
  = review-oriented CAD workflow
```

## 2. Image parsing and case tracking

Uploaded files are read by the API and routed through an image parser.

- Common image formats are opened as PIL images.
- DICOM-like inputs are detected and converted into PIL images.
- Available DICOM metadata is preserved in the response payload.
- A case ID is generated from the uploaded image bytes so each result can be tracked.

## 3. Prediction logic

### 3-1. Model mode selection

The API accepts a model key through the `model` query parameter:

```text
ensemble | densenet | efficientnet | vit
```

The selected model determines which loaded checkpoint or placeholder predictor is used.

### 3-2. Checkpoint detection

At server startup, the API searches for trained checkpoint files under:

```text
checkpoints/<model_key>/<model_key>_best.pth
checkpoints/<model_key>/*.pth
checkpoints/<model_key>_best.pth
```

If a checkpoint is found, the model is loaded for real inference. If not, the registry keeps that model in placeholder mode.

### 3-3. Placeholder mode

Placeholder mode exists to keep the interface demonstrable when no `.pth` file is present. Responses clearly include `Is_Placeholder=true`, so a demo does not misrepresent simulated output as real inference.

### 3-4. Ensemble logic

The ensemble mode averages probabilities from available model outputs. If a model is in placeholder mode, the response remains marked so the user can tell whether the result came from real weights or simulated values.

## 4. Threshold and detected-disease logic

Each disease probability is compared with the selected detection threshold.

```text
Detected_Diseases = labels where probability >= threshold
Top_Disease = label with highest probability
Top_Probability = maximum disease probability
```

The default threshold is configured as `0.3` in the API. The dashboard lets reviewers change the threshold interactively.

## 5. Report draft logic

The prediction response includes report-oriented fields:

- `Report_Draft`,
- `Findings_KR`,
- `Impression_KR`,
- `Need_Review_Reason`,
- `Clinical_Report`.

The draft is generated from predicted labels, top probability, threshold status, and review context. It is meant to be copied and edited by clinicians, not used as a final report.

## 6. Image quality logic

The agentic workflow computes deterministic image-quality indicators such as brightness, contrast, sharpness, and entropy-style signals. These values help the workflow flag whether the input may be difficult to interpret.

Quality checks are not model predictions. They are review aids that help answer: **is this image suitable for model-assisted interpretation?**

## 7. Anatomy and Grad-CAM logic

The workflow can attach Grad-CAM-style context and anatomy assessment to each case. The goal is to help reviewers inspect whether model attention appears consistent with plausible thoracic regions.

Important interpretation rule:

> A heatmap is supporting context, not proof that the model is correct.

Heatmap concerns can be sent to the feedback queue through the dashboard.

## 8. Triage assessment logic

The agent builds a triage-style assessment from top findings, probabilities, detected labels, and review signals. It helps prioritize cases inside the demo workflow but does not replace clinical triage.

Typical output includes:

- priority framing,
- review reason,
- detected disease summary,
- confidence-style language,
- and follow-up review suggestions.

## 9. Multi-case comparison logic

For batch review, the Agent Workbench constructs per-case rows and probability matrices. It can compare:

- top disease per image,
- top probability,
- number of detected labels,
- disease overlap across cases,
- and high-priority cases.

This supports questions such as “Which image should be reviewed first?” or “Which cases show similar disease patterns?”

## 10. Reliability readiness logic

Reliability Readiness combines multiple metrics into a pass/warning/critical status.

Main checks include:

- calibration ECE,
- Youden’s J operating-point quality,
- subgroup AUROC gap,
- external validation AUROC drop,
- shortcut-pattern ratio,
- ROI consistency or outside-ROI explanation risk,
- and hidden-strata warnings.

A critical issue blocks deployment in the readiness summary. Multiple warning issues can also escalate the status.

## 11. Feedback queue logic

Clinician feedback is saved as JSONL records. The queue item includes:

- generated queue ID,
- case ID,
- selected feedback type,
- corrected labels when provided,
- clinician comments,
- edited report text,
- review status,
- retraining-candidate flag,
- and a regulatory note.

The queue does not automatically update the model. It is intentionally a review and curation layer.

## 12. LLM and agent answer logic

When LLM settings are configured, follow-up chat answers are generated from a compact tool-context payload. The agent uses the current case or batch output as grounding context.

When LLM settings are unavailable, deterministic fallback answers are used for supported question types. This keeps the demo functional and reduces silent failure.

## 13. Safety boundary

The system should be interpreted as a research and workflow prototype. It can assist review, comparison, and documentation, but final interpretation must remain with qualified clinicians.
