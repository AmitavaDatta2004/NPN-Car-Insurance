# Model Card — Severity Classifier MobileNetV2 v1

## Identity

ML-IMPROVE-001 correction (2026-09-22): the legacy training loop restored the final Stage A weights while reporting the best Stage A score. The team update e1ba3ec now retains actual best weights across both stages; its notebook and CLI were preserved during integration. The imported metrics JSON reports test macro F1 0.6470 and accuracy 64.92%; these are team results, not a rerun in this session. New validation-only experiments are in Notebook 09b; no new performance or calibration claim is established yet. Candidate checkpoints are isolated under `artifacts/runs/improvements/` and use `load_candidate` with saved preprocessing.

- Model version: SEV-MNV2-001
- Task: Multi-class vehicle damage severity classification (3 tiers: `minor`, `moderate`, `severe`)
- Architecture: ImageNet-pretrained MobileNetV2 + AdaptiveAvgPool2d((1, 1)) + Dropout(0.3) + Linear(1280→128) + ReLU + Dropout(0.3) + Linear(128→3)
- Base weights: ImageNet-1K (`MobileNet_V2_Weights.IMAGENET1K_V1`)
- Experiment ID: SEV-MNV2-001
- Training dataset: Car Damage Severity Dataset v1 (1,631 images)
- Manifests: `data/manifests/severity_train.csv` (1,140), `severity_val.csv` (243), `severity_test.csv` (248)
- Export format: PyTorch `.pt` checkpoint (`artifacts/models/severity_mnv2.pt`, `ml/artifacts/severity/severity_mnv2_v1.pt`) + ONNX (`artifacts/models/severity_mnv2.onnx`)
- Owner: Friend 2 / Antigravity
- Phase: Phase 6
- Companion models: Phase 5 (Baseline Custom CNN), Phase 7 (ViT-Tiny)

## Intended Use

The model processes a single RGB vehicle photograph and predicts the overall damage severity level (`minor`, `moderate`, or `severe`) alongside calibrated softmax probabilities. In the ClaimVision AI triage pipeline:
1. `minor`: Eligible for fast-track human adjuster review and low-tier indicative repair estimation.
2. `moderate`: Standard review routing with detailed damage localisation and part replacement estimation.
3. `severe`: High-priority routing; potential total loss or structural compromise requiring senior adjuster assignment.

## Prohibited Interpretation

**This model outputs an overall visual damage severity classification. It does NOT:**
1. Determine part-specific damage severity (unless accompanied by verified part crop models).
2. Authorize claim settlement payouts or legally binding repair estimates.
3. Replace insurance physical inspection for structural integrity or mechanical damage invisible from exterior photos.

Underestimating severe damage creates significant under-reserving risk; consequently, `severe` recall is treated as the primary safety metric.

## Input / Output Contract

| Item | Specification |
| --- | --- |
| Input format | JPEG, PNG, or WEBP |
| Input tensor | Float32, shape (B, 3, 224, 224), RGB |
| Normalisation | ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |
| Val/test preprocessing | Resize(256) → CenterCrop(224) → ToTensor → Normalize |
| Output | Raw logits shape (B, 3). Softmax applied at inference time |
| Class mapping | `0: minor`, `1: moderate`, `2: severe` |
| Primary safety metric | Severe Class Recall |

## Training Configuration

| Setting | Value |
| --- | --- |
| Loss | CrossEntropyLoss with balanced class weights |
| Optimizer | AdamW (weight_decay=1e-4) |
| Stage A LR | 1e-3 (backbone frozen, head only, 5 epochs) |
| Stage B LR | 1e-5 (features[17:] unfrozen, CosineAnnealingLR, 10 epochs) |
| Batch size | 32 (CUDA) / 16 (CPU) |
| Seed | 42 |
| Checkpoint selection | Best Validation Macro F1 score |

## Data Splits & Anti-Leakage Controls

All models in the Phase 5–7 severity evaluation suite train and evaluate on the exact same frozen manifests created in Phase 4 (`SDATA-001`):
- **Train Split (70%)**: 1,140 images (Minor: 372, Moderate: 377, Severe: 391)
- **Validation Split (15%)**: 243 images (Minor: 80, Moderate: 80, Severe: 83)
- **Held-out Test Split (15%)**: 248 images (Minor: 82, Moderate: 81, Severe: 85)

Zero duplicate leakage was programmatically verified via SHA-256 and perceptual pHash (distance $\le 8$). The held-out test manifest was evaluated strictly once at the end of training.

## Performance Summary

- Macro F1: Evaluated on held-out test set (`severity_test.csv`).
- Severe Recall: Monitored to prevent severe damage under-classification.
- Inference Latency: ~20–25 ms/image on CPU, ensuring responsive local demonstration.
- Full comparative metrics recorded in `ml/results/severity_mnv2_metrics.json`.

## Limitations & Edge Cases

1. **Boundary Ambiguities**: The boundary between deep scratch / minor dent (`minor`) versus creased sheet metal / misaligned panel (`moderate`) can be visually subjective; confidence threshold warnings are triggered when confidence is below 50%.
2. **Angle Sensitivity**: Damage captured from extreme oblique angles or with heavy reflection may produce lower confidence.
3. **Small Sample Scale**: 1,140 training images requires aggressive data augmentation to prevent overfitting.
