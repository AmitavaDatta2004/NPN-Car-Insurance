# Notebook Review Checklist

Notebook:  
Owner:  
Reviewer:  
Task/experiment ID:  
Commit:

## Reproducibility

- [ ] Title, purpose, owner, date, dataset version, and task ID are present.
- [ ] Seeds and dependency versions are printed.
- [ ] Paths are repository-relative or configuration-based.
- [ ] Notebook runs after kernel restart using Run All.
- [ ] Training can be skipped when a verified artifact already exists.
- [ ] Artifact paths and checksums are shown.

## Scientific integrity

- [ ] Split leakage and duplicates are checked.
- [ ] Validation data—not test data—drives tuning and selection.
- [ ] Metrics match the problem and class imbalance.
- [ ] Successes and failures are both displayed.
- [ ] Claims are no stronger than the available labels.
- [ ] Limitations and uncertainty are explicit.

## OpenCV evidence

- [ ] Original image is retained and shown.
- [ ] Original/transformed comparison is visible.
- [ ] Operation and parameters are stated.
- [ ] Cropping/denoising/enhancement has an ablation before default use.
- [ ] Bounding boxes, labels, confidences, and legends are readable.

## Judge readability

- [ ] Markdown explains why each major step exists.
- [ ] Plots have title, axes, units, and legend.
- [ ] Tables are concise and readable at presentation scale.
- [ ] No excessive logs or progress spam remain.
- [ ] Final section states findings, limitations, and next decision.

## Safety

- [ ] No token, credential, personal path, face/plate leak, or private data appears.
- [ ] No huge embedded output or archive is committed.
- [ ] Notebook checkpoint directory is excluded.

## Review result

- Decision: ACCEPT / CHANGES REQUIRED
- Required changes:
- Reviewer/date:

