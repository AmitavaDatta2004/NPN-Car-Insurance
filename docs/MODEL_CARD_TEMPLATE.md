# Model Card — MODEL_NAME

## Identity

- Model version:
- Task:
- Architecture:
- Base weights:
- Experiment ID:
- Git commit:
- Training dataset/manifests:
- Export format:
- Artifact checksum:
- Owner/reviewer:

## Intended use

Describe supported inputs, outputs, and human-review role.

## Prohibited interpretation

State what the model cannot establish. Fraud-risk output must not be described as legal proof of fraud; severity must not be described as a final insurer decision.

## Input/output contract

| Item | Specification |
| --- | --- |
| Input format |  |
| Image size/color order |  |
| Normalization |  |
| Output labels |  |
| Thresholds |  |
| Confidence/calibration |  |

## Training configuration

- Loss:
- Optimizer/scheduler:
- Epochs and early stopping:
- Batch size:
- Augmentations:
- Seed:
- Hardware and duration:

## Evaluation

| Dataset split | Metric | Result |
| --- | --- | --- |
| Validation |  |  |
| Untouched test |  |  |

Include class-wise performance, confusion matrix, calibration, latency, size, and comparison with the baseline.

## Threshold decision

Document how the threshold was selected using validation data, the business trade-off, and what happens in the uncertainty band.

## Failure modes

List observed failures involving lighting, reflections, small damage, heavy occlusion, unusual vehicles, watermarks, edited images, or domain shift.

## Ethical and operational limitations

Document bias, human oversight, evidence retention, privacy, and why the output is advisory.

## Runtime integration

- Loader module:
- Prediction function:
- API endpoint:
- Model/version fields returned:
- Smoke test command:
- Rollback artifact:

## Approval

- Selected for prototype: Yes / No
- Approved by:
- Date:
- Revisit trigger:

