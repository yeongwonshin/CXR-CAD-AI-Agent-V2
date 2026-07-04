# Product Differentiation

## 1. What problem this project addresses

Many chest X-ray AI projects are technically impressive but remain difficult to present as an end-to-end product because they stop at one of the following outputs:

- a model training notebook,
- a metric table,
- a single probability vector,
- a Grad-CAM image,
- or a generic chatbot answer.

CXR-CAD AI Agent is structured as a review system around the model output. It shows how a computer-aided diagnosis pipeline can combine inference, explanation, reporting, agentic case review, reliability checks, and clinician feedback.

## 2. Difference from a classification notebook

| Notebook-only project | Limitation | CXR-CAD AI Agent |
| --- | --- | --- |
| Training and validation code is available | Hard to operate during a live review session | Adds FastAPI endpoints, Streamlit pages, upload workflow, and feedback capture |
| Metrics are shown in static cells | Reviewers must infer operational meaning | Presents operating points, calibration, subgroup gaps, external validation, and readiness status |
| Model checkpoints are separate from the demo | Demo may fail or silently use mock output | Explicitly detects checkpoint availability and marks placeholder mode |

## 3. Difference from a probability-only CAD demo

A probability-only demo answers: **what did the model predict?**

This project also asks:

- Is the input image usable?
- Was the input a DICOM file, and what metadata is available?
- Which findings crossed the threshold?
- What is the top finding and review reason?
- Can the probability output be translated into an editable report draft?
- Does the heatmap point to plausible anatomy?
- Should this case be reviewed before others?
- Did a clinician disagree, correct labels, or flag the heatmap?

## 4. Difference from a heatmap viewer

Heatmaps are useful but can be misleading when interpreted alone. This project treats Grad-CAM as one component of a broader review loop.

| Heatmap-only viewer | Limitation | CXR-CAD AI Agent |
| --- | --- | --- |
| Displays attention overlay | Does not explain whether the focus is clinically plausible | Adds anatomy assessment and shortcut-pattern review |
| No deployment gate | Weak localization may be ignored | Adds ROI consistency and readiness checks |
| No feedback loop | Review issues are not captured | Saves heatmap concerns and corrected labels into a feedback queue |

## 5. Difference from generic medical LLM demos

A prompt-only medical chatbot can sound fluent while being disconnected from the actual image model output. CXR-CAD AI Agent keeps follow-up answers grounded in the current case payload.

The agent can use:

- predicted probabilities,
- detected diseases,
- case IDs and filenames,
- image-quality indicators,
- DICOM metadata,
- report draft fields,
- Grad-CAM availability,
- triage and anatomy assessments,
- cross-case comparison tables,
- and previous chat history.

When LLM settings are not configured, the app falls back to deterministic context-based responses instead of failing silently.

## 6. Difference from a generic dashboard

Generic dashboards show charts. This project maps each chart to a review decision.

| Dashboard area | Decision supported |
| --- | --- |
| Prediction dashboard | Which findings need review for this image? |
| Agent Workbench | Which case in a batch should be reviewed first, and why? |
| Analysis Results | Which model behavior is strong or weak across thresholds and subgroups? |
| Reliability Readiness | Is the model ready for broader deployment, limited deployment, or blocked review? |
| Feedback Queue | Which clinician corrections should become curated review artifacts? |

## 7. Demo strengths

The strongest points to show in a demo are:

1. upload an image or DICOM file,
2. show predicted findings and report draft,
3. show Grad-CAM and quality/anatomy context,
4. run Agent Workbench on multiple files,
5. ask a follow-up question grounded in the case payload,
6. open Reliability Readiness and show why deployment is pass, warning, or critical,
7. submit clinician feedback and show it in the queue.

## 8. Important limitation

The project demonstrates workflow design around CAD inference. It is not a medical device and should not be used for real diagnosis without separate clinical validation, regulatory review, and qualified clinician oversight.
