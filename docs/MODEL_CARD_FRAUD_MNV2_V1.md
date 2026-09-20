# Model Card — Fraud Classifier MobileNetV2 v1

## Identity

- Model version: FRAUD-MNV2-001
- Task: Binary visual fraud-risk classification (genuine vs. suspicious)
- Architecture: MobileNetV2 backbone + Dropout + Linear(1280→128) + ReLU + Dropout + Linear(128→1)
- Base weights: ImageNet-pretrained MobileNetV2 (torchvision `IMAGENET1K_V1`)
- Experiment ID: FRAUD-MNV2-001
- Training dataset: Vinay Jose Car Damage Dataset v1 (8,079 images)
- Manifests: notebooks/data/manifests/fraud_train.csv, _val.csv, _test.csv
- Export format: PyTorch `.pt` checkpoint + ONNX (opset 17)
- Owner: Member 2 / Antigravity
- Phase: 2

## Intended Use

The model receives a single RGB vehicle photograph and returns a fraud-risk
probability. This probability is used as one signal in the ClaimVision AI
triage pipeline to decide whether to route a claim to human fraud review or
continue to automated damage assessment.

## Prohibited Interpretation

**This model outputs a visual fraud-risk probability. It does NOT confirm
insurance fraud. A high score indicates suspicious image characteristics
relative to the training distribution. It does not establish intent, prove
manipulation, or support claim rejection without human investigation.**

High risk → human fraud review. The system never auto-rejects a claim
based on this model's output.

## Input / Output Contract

| Item | Specification |
| --- | --- |
| Input format | JPEG, PNG, or WEBP |
| Input tensor | Float32, shape (B, 3, 224, 224), RGB |
| Normalisation | ImageNet: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |
| Val/test preprocessing | Resize(256) → CenterCrop(224) → ToTensor → Normalize |
| Output | Raw logit shape (B, 1). Sigmoid applied at inference time |
| Probability | float [0.0, 1.0]. Higher = more suspicious |
| Labels | 0 = genuine, 1 = suspicious |
| Thresholds | Loaded from `ml/artifacts/fraud/thresholds_v1.json` |

## Training Configuration

| Setting | Value |
| --- | --- |
| Loss | BCEWithLogitsLoss with pos_weight (class-frequency inverse) |
| Optimizer | AdamW |
| Stage A LR | 1e-3 (frozen backbone, head only) |
| Stage B LR | 1e-5 (final 2 InvertedResidual blocks unfrozen) |
| Batch size | 16 |
| Imbalance handling | WeightedRandomSampler + pos_weight |
| Max epochs | Stage A: 10, Stage B: 20 |
| Early stopping | patience=5 on val PR-AUC |
| Seed | 42 |
| Device | CPU |

## Evaluation

Metrics recorded after thresholds were frozen. Test set used exactly once.

| Split | Metric | Result |
| --- | --- | --- |
| Validation | PR-AUC | 0.4999 |
| Validation | ROC-AUC | 0.9108 |
| Test | PR-AUC | 0.5464 |
| Test | ROC-AUC | 0.9077 |
| Test | Suspicious Recall | 0.9014 (64 / 71) |
| Test | Suspicious Precision | 0.1400 (64 / 457) |
| Test | Suspicious F1 | 0.2424 |
| Test | Confusion Matrix | TN=750, FP=393, FN=7, TP=64 (threshold=0.80) |
| Inference | CPU Latency | 22.60 ms/image (+/- 0.74 ms) |

## Threshold Decision

Two routing thresholds are stored in `ml/artifacts/fraud/thresholds_v1.json`:

- `low_threshold`: Below this probability, image is considered low risk.
- `high_threshold`: At or above this probability, image is high risk → FRAUD_REVIEW.

Thresholds were chosen by sweeping the validation precision-recall curve
aiming for suspicious recall >= 0.80. The test set was not consulted.

Business rationale: The cost of missing a suspicious image (false negative)
is higher than the cost of routing a genuine image to human review (false positive).

## Failure Modes

1. **Shortcut learning:** The model may classify based on image resolution,
   watermarks, compression artefacts, or source domain rather than genuine fraud signals.
2. **Small positive class:** Only 325 training suspicious examples. Wide confidence intervals.
3. **Domain mismatch:** Images not representative of Indian vehicles, lighting, or damage.
4. **Adversarial:** A sophisticated actor could submit a suspicious image that has been
   processed to look like a genuine image.

## Ethical and Operational Limitations

- Output is advisory only. Human reviewers make all final decisions.
- High-risk routing increases reviewer workload; tune thresholds with real operational data.
- Absence of EXIF data alone does NOT create a fraud signal (per AGENTS.md rule).
- Model must not be described as detecting insurance fraud in any user-facing copy.

## Runtime Integration

- Loader module: `claimvision_ml.fraud.model.load_fraud_model`
- Prediction function: `claimvision_ml.fraud.predict_fraud`
- API endpoint (Phase 13): `POST /api/v1/claims/{id}/assess`
- Model version field: `model_version: "FRAUD-MNV2-001"`
- Smoke test: `pytest ml/tests/test_fraud_model.py -v`

## Approval

- Selected for prototype: Yes (pending training completion)
- Approved by: Team Lead
- Date: 2026-09-20
- Revisit trigger: If shortcut audit finds resolution/watermark dominates (>80%
  feature importance), label the classifier experimental and rely on deterministic
  evidence-integrity signals instead.
