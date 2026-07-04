# Clinical Reliability

## 1. Purpose

Clinical Reliability summarizes whether the current model artifacts look acceptable for broader review, limited review, or blocked deployment. It is designed as a pre-deployment checklist rather than a regulatory claim.

## 2. Readiness dimensions

| Dimension | Example signal | Why it matters |
| --- | --- | --- |
| Calibration | Expected Calibration Error | A probability should roughly match observed frequency |
| Operating point | Youden’s J, sensitivity, specificity | Thresholds should match the use case, such as screening or diagnostic support |
| Domain robustness | subgroup AUROC gaps, external validation drop | Performance should not collapse for view position, age group, gender, or outside data |
| Localization | shortcut ratio, ROI consistency | Explanations should not rely heavily on markers, devices, text, or background artifacts |
| Hidden stratification | underperforming clusters | A model can perform well on average while failing on hidden subgroups |

## 3. Default readiness rules

The readiness builder and dashboard use adjustable thresholds. Default values include:

| Check | Default rule | Status impact |
| --- | --- | --- |
| ECE | warning when ECE is at least 0.05 | calibration warning |
| Youden’s J | warning when below 0.60 | operating-point warning |
| subgroup gap | warning when AUROC gap is at least 3.0 percentage points | domain robustness warning |
| external validation drop | critical when AUROC drop is at least 3.0 percentage points | domain robustness critical |
| shortcut ratio | warning when above the selected threshold | localization warning |
| outside-ROI explanation energy | critical when outside-ROI ratio is at least 40% | localization critical |
| hidden strata | warning when one or more underperforming strata are detected | hidden-stratification warning |

The dashboard can escalate status to **CRITICAL** when a critical issue exists or when too many warnings accumulate.

## 4. Status interpretation

| Status | Meaning | Recommended action |
| --- | --- | --- |
| PASS | No active warning or critical issue | Proceed with routine monitoring in the demo context |
| WARNING | One or more review issues exist | Review the affected metric, subgroup, threshold, or explanation pattern before broader use |
| CRITICAL | A blocking issue exists | Do not deploy until the issue is resolved and revalidated |

## 5. Calibration review

Calibration checks whether predicted probabilities are numerically meaningful. ECE is used as a compact signal. If calibration is weak, the project recommends temperature scaling and threshold reset before using probability values operationally.

## 6. Operating-point review

Operating points are disease-specific. A screening-oriented setting may prioritize sensitivity, while a diagnostic-support setting may require higher specificity. The dashboard and model evaluation document include examples of threshold trade-offs.

## 7. Domain robustness review

The project compares internal validation behavior with external validation artifacts and subgroup results. Common review slices include:

- gender,
- age group,
- view position,
- and external dataset performance.

A high gap indicates the need for additional validation, reweighting, fine-tuning, domain adaptation, or slice-specific monitoring.

## 8. Localization and shortcut review

Grad-CAM and shortcut-region outputs are used to inspect whether attention appears clinically plausible. Suspicious patterns include excessive attention on:

- text markers,
- pacemakers or devices,
- ribs or clavicles unrelated to the target finding,
- image borders,
- and background artifacts.

## 9. ROI consistency extension

The reliability module includes ROI consistency scoring. It measures how much explanation energy lies inside the valid region of interest. A high outside-ROI ratio is treated as a localization risk.

## 10. Hidden stratification extension

Hidden-stratification detection clusters embeddings and looks for underperforming strata. This helps identify failure pockets that may not be visible in broad subgroup tables.

## 11. Safety note

Readiness status is a demonstration aid. It is not regulatory clearance, clinical validation, or proof of safety. Real deployment would require formal clinical validation, governance, monitoring, and regulatory review.
