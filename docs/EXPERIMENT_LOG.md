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
| FRAUD-MNV2-001 | 2026-09-20 | Member 2 / Antigravity | de7100c | Vinay Jose v1 / fraud_train.csv, _val.csv, _test.csv | MobileNetV2 (ImageNet pretrained) | 42 | COMPLETE | Val PR-AUC: 0.4999 / Test PR-AUC: 0.5464 | ml/artifacts/fraud/fraud_mnv2_v1.pt | ACCEPTED (Phase 2 Gate passed) |

## Detailed experiment entries

### FRAUD-MNV2-001 — Fraud MobileNetV2 Baseline

- Task ID: ML-001
- Owner/reviewer: Member 2 / Antigravity | Reviewer: Member 1
- Start time: 2026-09-20 19:27 IST
- End time: TBD (after Notebook 02 training completes)
- Git commit: de7100c
- Notebook: notebooks/02_fraud_mobilenetv2_training.ipynb
- Evaluation notebook: notebooks/03_fraud_evaluation_and_threshold.ipynb
- Dataset card: docs/DATASET_CARD_FRAUD.md
- Dataset version: Vinay Jose Car Damage v1 (8,079 images)
- Manifest version: fraud_train.csv / _val.csv / _test.csv (70/15/15 split, seed=42)
- Hardware: CPU (local)
- Software: torch>=2.2, torchvision>=0.17, Python 3.11.9
- Seed: 42
- Hypothesis: A fine-tuned MobileNetV2 can learn visual fraud-risk signals achieving PR-AUC above the naive baseline (class ratio).

Configuration:

```yaml
model: mobilenet_v2
pretrained_weights: ImageNet (torchvision IMAGENET1K_V1)
image_size: [224, 224]
batch_size: 16
stage_a:
  optimizer: AdamW
  learning_rate: 1.0e-3
  epochs: 10
  frozen: all backbone
stage_b:
  optimizer: AdamW
  learning_rate: 1.0e-5
  epochs: 20
  unfrozen: features[17:]  (last 2 InvertedResidual blocks)
loss: BCEWithLogitsLoss with pos_weight (n_genuine / n_suspicious)
augmentations:
  - RandomResizedCrop(224, scale=(0.7, 1.0))
  - RandomHorizontalFlip(p=0.5)
  - ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
  - RandomRotation(degrees=15)
early_stopping: patience=5 on val PR-AUC
threshold_method: validation PR curve sweep (0.20-0.80)
imbalance: WeightedRandomSampler + BCEWithLogitsLoss pos_weight
class_counts_train: genuine=5329, suspicious=325 (ratio 16.4:1)
```

Results:

| Split | Metric | Value |
| --- | --- | --- |
| Validation | PR-AUC | 0.4999 |
| Validation | ROC-AUC | 0.9108 |
| Test | PR-AUC | 0.5464 |
| Test | ROC-AUC | 0.9077 |
| Test | Suspicious Recall | 0.9014 (64 / 71) |
| Test | Suspicious Precision | 0.1400 (64 / 457) |
| Test | Suspicious F1 | 0.2424 |
| Test | Confusion Matrix | TN=750, FP=393, FN=7, TP=64 |
| Inference | CPU Latency | 22.60 ms/image (+/- 0.74 ms) |

Artifacts:

- Checkpoint: ml/artifacts/fraud/fraud_mnv2_v1.pt (gitignored)
- ONNX: ml/artifacts/fraud/fraud_mnv2_v1.onnx (gitignored)
- Thresholds: ml/artifacts/fraud/thresholds_v1.json (committed)
- Preprocessing config: ml/artifacts/fraud/preprocessing_config.json (committed)
- Class map: ml/artifacts/fraud/class_map.json (committed)
- Training history: ml/artifacts/fraud/training_history.json (committed)
- Plots: ml/results/fraud/*.png (committed, 11 plots)

Conclusion:

Hypothesis supported: MobileNetV2 fine-tuned with class weighting and weighted sampling
achieves 0.5464 PR-AUC and 0.9077 ROC-AUC on the held-out test set, well above the naive
baseline (0.058 class ratio).
At the frozen high threshold (0.80), the model achieves 90.1% recall on suspicious claims
(catching 64 out of 71 suspicious images) with 393 false positives routed to human fraud review.
CPU inference latency is 22.60 ms/image. Standalone `predict_fraud` verified end-to-end.
Phase 2 exit gate passed. Ready for Phase 3 (OpenCV Evidence Integrity).


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

