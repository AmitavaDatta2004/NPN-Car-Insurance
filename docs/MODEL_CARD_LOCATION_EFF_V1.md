# Model Card — Location EfficientNet-B0 v1

**Experiment ID:** LOC-EFF-001  
**Task:** Image-level damaged-part location classification  
**Architecture:** EfficientNet-B0 (ImageNet pretrained via `timm`, 2-stage fine-tuning)  
**Status:** PENDING training — fill in metrics after Notebook 14 is executed in Colab  
**Owner:** Location CNN member / Antigravity  
**Notebook:** `notebooks/14_location_efficientnet_training.ipynb`  
**Date:** 2026-09-22

---

## Model Description

**What it does:** Given a full car damage photograph, predict which vehicle part is visibly damaged.  
Output is one of five image-level class labels (no bounding box drawn).

| Class ID | Class Name |
|---|---|
| 0 | headlamp |
| 1 | front_bumper |
| 2 | hood |
| 3 | door |
| 4 | rear_bumper |

**What it does NOT do:** Localise the damage with a bounding box. Use the YOLO part detector (Notebook 12 / `parts.py`) for bounding-box predictions.

---

## Architecture

```
224 × 224 RGB image
  → EfficientNet-B0 backbone (timm, ImageNet1K pretrained, 8 MBConv blocks)
  → global average pooling (built into timm model with global_pool="avg")
  → Dropout(0.3)
  → Linear(1280 → 128)
  → ReLU
  → Dropout(0.3)
  → Linear(128 → 5)  [logits; apply softmax for probabilities]
```

**Why EfficientNet-B0?**  
Compound scaling (depth × width × resolution) provides better accuracy than MobileNetV2 on most benchmarks with only a modest parameter increase (5.3 M vs 3.4 M). Available via `timm`, already a project dependency from ViT-Tiny severity experiments.

**Training strategy:**

| Stage | Frozen | Epochs | LR |
|---|---|---|---|
| A (head only) | Entire EfficientNet backbone | 25 (patience 10) | 1e-3 |
| B (fine-tune) | Blocks 0 – 5 (last 2 of 8 unfrozen) | 15 (patience 8) | 5e-5 |

Loss: `CrossEntropyLoss` with inverse-frequency class weights.  
Optimizer: AdamW with weight decay 1e-4 and CosineAnnealingLR.

---

## Dataset

See `docs/DATASET_CARD_LOCATION.md` for full details.

| Split | Images | Labels |
|---|---|---|
| Train | 59 | Derived (dominant-part rule) |
| Val | 11 | Derived (dominant-part rule) |
| Test | 8 | None (unannotated, visual inspection only) |

> **Critical constraint:** ~12 training images per class. Results are prototype-quality only.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Seed | 42 |
| Image size | 224 × 224 |
| Normalisation | ImageNet (mean=[0.485, 0.456, 0.406]; std=[0.229, 0.224, 0.225]) |
| Augmentation | Random resized crop, horizontal flip, colour jitter, ±15° rotation |
| Batch size | 8 |
| Hardware | Google Colab GPU |

---

## Results

> **Fill in after Notebook 14 is executed in Colab.**

### Validation Metrics (val split, 11 images)

| Metric | Value |
|---|---|
| Accuracy | TBD |
| Macro Precision | TBD |
| Macro Recall | TBD |
| Macro F1 | TBD |
| headlamp F1 | TBD |
| front_bumper F1 | TBD |
| hood F1 | TBD |
| door F1 | TBD |
| rear_bumper F1 | TBD |

### Benchmark

| Metric | Value |
|---|---|
| Model file size (`.pt`) | TBD MB |
| CPU latency (local, 25 runs avg) | TBD ms |
| ONNX exported | TBD |

---

## Artifacts

| Artifact | Path |
|---|---|
| PyTorch checkpoint | `ml/results/location/location_efficientnet_best.pt` |
| Final checkpoint (copied) | `artifacts/models/location_efficientnet.pt` |
| ONNX model | `artifacts/models/location_efficientnet.onnx` |
| Stage A training curves | `ml/results/location/location_efficientnet_stage_a_curves.png` |
| Stage B training curves | `ml/results/location/location_efficientnet_stage_b_curves.png` |
| Validation confusion matrix | `ml/results/location/location_efficientnet_val_cm.png` |
| Test predictions plot | `ml/results/location/location_efficientnet_test_predictions.png` |

---

## Intended Use

- Judge demonstration of part-level damage classification within the ClaimVision AI prototype.
- Comparison with MobileNetV2 (LOC-MNV2-001) in Notebook 15.
- The winning model from comparison is integrated into Notebook 16 (Unified Inference Demo).

## Out of Scope

- Production insurance claim processing.
- High-confidence identification of specific damage types.
- Images with severe occlusion or non-standard damage patterns.

## Limitations

- Dataset too small (~12 images/class) for reliable generalisation.
- EfficientNet-B0 has more parameters than MobileNetV2 — may overfit more aggressively on tiny dataset.
- Front vs. rear bumper confusion expected due to visual similarity.
- `timm` version must match training environment when loading checkpoint for inference.

## Ethical Considerations

This model provides a visual signal about which vehicle part appears damaged. It does not determine insurance liability, fault, or payout. Outputs are supplementary evidence for human review — not decisions.

---

## Comparison Status

Compare with `docs/MODEL_CARD_LOCATION_MNV2_V1.md` using Notebook 15 results.

Selected model: **TBD** (fill in after Notebook 15)
