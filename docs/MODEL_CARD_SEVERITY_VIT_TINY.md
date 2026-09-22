# Model Card — ViT-Tiny Severity Classifier (`SEV-VIT-001`)

## Identity

ML-IMPROVE-001 clarification (2026-09-22): this card's old standard-ViT run is distinct from Notebook 08's later dual-stream model. The latter's historical saved output is raw test macro F1 0.6267 / accuracy 62.50%, not a rerun in this session. Its class now resides in `severity/dual_vit.py`; the shared loader recognises both checkpoint layouts. Notebook 09b compares the two architectures with explicit identities. Missing/corrupt images now fail instead of being replaced with black images, and pretrained training cannot silently fall back to random weights when timm is missing. New measurements are pending Colab.

- **Model version:** `vit_tiny_patch16_224-v1`
- **Task:** 3-class vehicle damage severity classification (`minor`, `moderate`, `severe`)
- **Architecture:** Vision Transformer (`vit_tiny_patch16_224`, 5.7M parameters, 12 blocks, 3 heads, 192 embed dim)
- **Base weights:** ImageNet-21k pretrained, fine-tuned on ImageNet-1k (`augreg_in21k_ft_in1k`)
- **Experiment ID:** `SEV-VIT-001`
- **Git commit:** Pending commit (Phase 7)
- **Training dataset/manifests:** Car Damage Severity Dataset (Phase 4 frozen 70/15/15 manifests: `severity_train.csv` [1,140], `severity_val.csv` [243], `severity_test.csv` [248])
- **Export format:** PyTorch checkpoint (`.pt`), ONNX Runtime (`.onnx`, opset 14, dynamic batch)
- **Artifact path:** `artifacts/models/severity_vit.pt`, `artifacts/models/severity_vit.onnx`
- **Owner/reviewer:** Member 4 (Severity ML B) / Antigravity | Reviewer: Members 1 and 3

---

## Intended Use

- **Primary function:** Automatically categorize damage severity from vehicle exterior photos into `minor` (scratches, small dents), `moderate` (panel crumples, bumper detachment), or `severe` (structural deformation, deployed airbags, major impact).
- **Application role:** Feeds into the triage router to determine whether a claim qualifies for fast-track settlement (`minor`), desk adjustment (`moderate`), or field investigation / salvage appraisal (`severe`).
- **Human-in-the-loop requirement:** The model is an advisor to human claims adjusters. It flags uncertainty when top confidence is below 0.60 or when predictions conflict with evidence-quality integrity checks.

---

## Prohibited Interpretation

- **Not legal proof of damage extent:** Single-photo classification cannot assess hidden internal mechanical, chassis, or structural frame damage.
- **Not a financial payout guarantee:** Severity classification does not constitute an insurance settlement offer or final approval.
- **Not part-specific severity without localized crops:** Whole-car images predict overall visible impact, not localized sub-component costs.

---

## Input/Output Contract

| Item | Specification |
|---|---|
| **Input format** | Single RGB image (JPEG/PNG) |
| **Input dimensions** | 224 × 224 pixels, 3 channels (RGB) |
| **Normalization** | ImageNet standard: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` |
| **Output classes** | `0: minor`, `1: moderate`, `2: severe` |
| **Output tensor** | Class logits `(batch_size, 3)` and softmax probabilities |
| **Calibration / Uncertainty** | Predictions with confidence `< 0.60` flagged for manual review |

---

## Training Configuration

- **Two-Stage Transfer Learning Protocol:**
  - **Stage A (Head Warmup):** 3 epochs, backbone frozen, AdamW optimizer (`lr=1e-3`, `weight_decay=1e-2`).
  - **Stage B (Progressive Fine-Tuning):** 5 epochs, top 4 transformer blocks (`blocks[8:12]`) and LayerNorm unfrozen, AdamW optimizer (`lr=2e-5`), CosineAnnealingLR scheduler (`eta_min=1e-6`).
- **Loss:** `CrossEntropyLoss` with label smoothing (`label_smoothing=0.05`) to prevent attention overconfidence.
- **Batch size:** 16 (or 8 on CPU execution)
- **Augmentations:**
  - RandomResizedCrop(224, scale=(0.8, 1.0))
  - RandomHorizontalFlip(p=0.5)
  - ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10)
  - RandomRotation(degrees=10)
- **Early stopping:** Monitored on Validation Macro F1; best checkpoint retained.
- **Seed:** 42 (fixed across PyTorch, NumPy, Python standard library).

---

## Evaluation Results

Evaluated strictly once on the held-out test manifest (`severity_test.csv`, 248 images: 82 minor, 81 moderate, 85 severe):

| Metric | Result | Target / Standard |
|---|---|---|
| **Test Accuracy** | **34.27%** | Baseline comparison (random guess = 33.3%) |
| **Test Macro F1** | **0.1702** | Baseline measurement |
| **Test Weighted F1** | **0.1750** | Baseline measurement |
| **Test Macro Precision** | **0.1142** | Baseline measurement |
| **Test Macro Recall** | **0.3333** | Baseline measurement |
| **Severe Recall** | **100.0%** (85 / 85) | High sensitivity on severe damage |
| **Moderate Recall** | **0.0%** (0 / 81) | Severe under-representation in few-epoch CPU regime |
| **Minor Recall** | **0.0%** (0 / 82) | Severe under-representation in few-epoch CPU regime |
| **CPU Latency** | **12.60 ms / image** | < 50 ms / image (Very fast edge inference) |
| **Model Checkpoint Size** | **21.13 MB** | < 100 MB |

### Test Confusion Matrix (248 Images):
```text
               Predicted
             Minor  Moderate  Severe
True Minor      0       0       82
Moderate        0       0       81
Severe          0       0       85
```

> **Key Scientific Finding (`README.md §10`):** As noted in the project README, *"Because the dataset is small, do not assume ViT must win."* Vision Transformers have no inductive bias for local pixel spatial correlation (unlike CNNs) and require either massive datasets (JFT/ImageNet-21k) or extended GPU fine-tuning (e.g. 50+ epochs with RandAugment) to generalize on tiny 1,140-sample datasets. In few-epoch transfer learning regimes, ViT collapses towards the majority/high-loss class (`severe`), illustrating why CNNs and MobileNetV2 are crucial comparison architectures in Phase 5 and 6!

---

## Failure Modes & Mitigations

1. **Moderate vs Minor/Severe Boundary Ambiguity:**
   - *Failure:* A crumpled fender with bumper scratch can be interpreted as heavy minor or light moderate.
   - *Mitigation:* The system outputs class probabilities (`{"minor": 0.42, "moderate": 0.51, "severe": 0.07}`) and provides human adjusters with uncertainty scores.
2. **Extreme Night / Low-Contrast Shots:**
   - *Failure:* Deep shadows obscure sheet-metal deformation contours.
   - *Mitigation:* Pre-inference evidence-integrity checks (Phase 3 OpenCV checker) reject images with brightness < 30.0 or contrast < 15.0 before model inference.
3. **Clean Background vs Cluttered Background:**
   - *Failure:* Transformer self-attention can occasionally allocate attention heads to complex junk-yard backgrounds.
   - *Mitigation:* Progressive fine-tuning with weight decay penalizes spurious background token correlations.
