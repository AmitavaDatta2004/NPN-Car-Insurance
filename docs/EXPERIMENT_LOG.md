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
| SEV-MNV2-001 | 2026-09-21 | Friend 2 / Antigravity | 28debd5 | Car Damage Severity v1 / severity_train.csv, _val.csv, _test.csv | MobileNetV2 (ImageNet pretrained) | 42 | COMPLETE | Two-stage transfer learning (Macro F1 & Severe recall prioritized) | artifacts/models/severity_mnv2.pt | READY_FOR_COMPARISON (Phase 6 / Notebook 09) |


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

### SEV-MNV2-001 — Severity MobileNetV2 Transfer Learning

- Task ID: SEV-MNV2-001
- Owner/reviewer: Friend 2 / Antigravity | Reviewer: Member 1, Member 3
- Start time: 2026-09-21 17:47 IST
- Git commit: 28debd5
- Notebook: notebooks/07_severity_mobilenetv2_training.ipynb
- CLI Runner: scripts/train_severity_mobilenet.py
- Dataset card: docs/DATASET_CARD_SEVERITY.md
- Model card: docs/MODEL_CARD_SEVERITY_MNV2_V1.md
- Dataset version: Car Damage Severity Dataset v1 (1,631 images)
- Manifest version: severity_train.csv (1,140), severity_val.csv (243), severity_test.csv (248) (70/15/15 split, seed=42)
- Hardware: Bimodal (Local CPU / Colab GPU)
- Software: torch>=2.2, torchvision>=0.17, Python 3.11+
- Seed: 42
- Hypothesis: An ImageNet-pretrained MobileNetV2 trained via a two-stage protocol (Stage A: head-only on frozen backbone; Stage B: low-learning-rate fine-tuning of top InvertedResidual blocks) will achieve strong calibration, high macro F1, and particularly high recall on the critical 'severe' damage class on the held-out test set, while maintaining ultra-low CPU inference latency (<30 ms/image).

Configuration:

```yaml
model: mobilenet_v2
pretrained_weights: ImageNet-1K (torchvision MobileNet_V2_Weights.IMAGENET1K_V1)
image_size: [224, 224]
batch_size: 32 (CUDA) / 16 (CPU)
stage_a:
  optimizer: AdamW
  learning_rate: 1.0e-3
  weight_decay: 1.0e-4
  epochs: 5
  frozen: all backbone features
stage_b:
  optimizer: AdamW
  learning_rate: 1.0e-5
  weight_decay: 1.0e-4
  scheduler: CosineAnnealingLR (T_max=10, eta_min=1e-7)
  epochs: 10
  unfrozen: features[17:] (last 2 InvertedResidual blocks)
loss: CrossEntropyLoss with balanced class weights
augmentations:
  - RandomResizedCrop(224, scale=(0.8, 1.0))
  - RandomHorizontalFlip(p=0.5)
  - ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
  - RandomRotation(degrees=15)
checkpoint_criteria: Best validation Macro F1 score
class_counts_train: minor=372, moderate=377, severe=391 (total=1,140)
```

Artifacts:

- Checkpoint: `artifacts/models/severity_mnv2.pt` & `ml/artifacts/severity/severity_mnv2_v1.pt` (gitignored)
- ONNX: `artifacts/models/severity_mnv2.onnx` (gitignored)
- Metrics summary: `ml/results/severity_mnv2_metrics.json` (committed)
- Plots: `ml/results/severity/mnv2_*.png` (committed)
- Standalone inference function: `claimvision_ml.severity.predict_severity`

Conclusion:

Two-stage transfer learning architecture verified and implemented with bimodal Colab/local execution support. Checkpointing adheres strictly to validation Macro F1 to prevent held-out test leakage. Model card and test suite fully verified with zero regressions across full ML package. Ready for parallel evaluation and comparison against Phase 5 (CNN) and Phase 7 (ViT-Tiny) in Notebook 09.
