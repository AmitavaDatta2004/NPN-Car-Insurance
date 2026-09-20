# ML Experiment Registry

Add an entry before training. Update it after evaluation. Never delete an unsuccessful experiment; failed experiments are evidence.

## Experiment ID convention

- Fraud: `FRAUD-ARCH-NNN`
- Severity: `SEV-ARCH-NNN`
- Damage detector: `DMG-YOLO-NNN`
- Part detector: `PART-YOLO-NNN`
- Unified pipeline: `PIPELINE-NNN`
- OpenCV ablation: `CV-OPERATION-NNN`

## Registry

| ID | Date | Owner | Git commit | Dataset/manifests | Model | Seed | Status | Primary result | Artifact path | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| — | — | — | — | — | — | — | Not started | — | — | — |

## Detailed experiment entry

### EXPERIMENT-ID — Title

- Task ID:
- Owner/reviewer:
- Start/end time:
- Git commit:
- Notebook:
- Dataset card/version/checksum:
- Manifest version:
- Hardware:
- Software versions:
- Seed:
- Hypothesis:

Configuration:

```yaml
model:
pretrained_weights:
image_size:
batch_size:
epochs:
optimizer:
learning_rate:
scheduler:
loss:
augmentations:
early_stopping:
threshold_method:
```

Results:

| Split | Metric | Value |
| --- | --- | --- |
| Validation | Primary metric | TBD |
| Test | Primary metric | TBD |

Artifacts:

- Checkpoint:
- Exported model:
- Metrics JSON:
- Plots:
- Prediction samples:

Conclusion:

State what was learned, whether the hypothesis was supported, observed failure modes, and the next action.

