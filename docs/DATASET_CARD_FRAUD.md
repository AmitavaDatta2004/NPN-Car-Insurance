# Dataset Card — Vinay Jose Car Damage Dataset (Fraud Signal Audit)

## Identity

- Dataset name: Vinay Jose Car Damage Dataset
- Internal version: `fraud-vinayjose-v1`
- Source URL: `https://www.kaggle.com/datasets/vinayjose/car-damage-dataset`
- Download method: `kagglehub.dataset_download("vinayjose/car-damage-dataset")`
- Download date: 2026-09-20
- Licence and permitted use: Open public Kaggle research dataset / Academic and demonstration use
- Original archive checksum: Recorded in audit manifest upon download
- Maintainer: Member 2 (Fraud ML)
- Related tasks/notebooks: `DATA-001`, `notebooks/01_fraud_dataset_audit.ipynb`

## Intended use

The dataset is audited to evaluate whether visual features support a calibrated suspicious-image indicator for the first stage of the ClaimVision AI pipeline.

**Crucial Product & Legal Boundaries**:
- This dataset models **visual claim fraud risk / suspicious image presentation**, not legal fraud.
- An image classified as suspicious does NOT confirm legal fraud, will NOT automatically deny a claim, and triggers manual human fraud review.
- Missing metadata or visual artifact flags must never be treated as definitive fraud evidence.

## Structure and schema

The raw dataset contains categorized vehicle damage photographs and metadata:

| Field/path | Type | Meaning | Required | Notes |
|---|---|---|---|---|
| `image_id` / `filename` | string | Relative file path or base name of image | Yes | Validated against disk |
| `label` | string / int | Target class (`genuine` vs `fraud` / `suspicious`) | Yes | Binarized for classifier |
| `claim_id` | string | Claim / incident identifier | No (synthetic grouping key if missing) | Used to enforce group-level isolation |
| `sha256` | string | SHA-256 byte-level exact hash | Generated | Detects exact duplicates |
| `dhash` | string | 64-bit perceptual difference hash | Generated | Detects near-duplicates |
| `blur_score` | float | Grayscale Laplacian variance | Generated | Detects blurry submissions |
| `brightness` | float | Mean pixel intensity (0–255) | Generated | Detects under/overexposure |
| `contrast` | float | Standard deviation of grayscale pixel intensity | Generated | Detects washed-out images |

## Label taxonomy

| Label | Operational definition | Scientific Interpretation |
|---|---|---|
| `genuine` / `0` | Vehicle damage photograph consistent with genuine claim evidence | Normal triage processing eligible for automated damage estimation |
| `fraud` / `suspicious` / `1` | Staged, manipulated, reused, or anomalous vehicle damage image | Suspicious visual signal — halts automatic processing, routes to manual review |

## Quality audit

- Total records/images: 8,079
- Decodable images: 8,079 / 8,079 (100% verified via OpenCV)
- Corrupt/missing: 0
- Exact duplicates (SHA-256): 0
- Near duplicates: 2,502 perceptual clusters identified via dHash ($\le 4$ Hamming distance)
- Class balance: Genuine (`0`): 7,614 (94.24%), Suspicious (`1`): 465 (5.76%)
- Resolution distribution: Audited; height and width distributions evaluated across classes
- Watermark / border / background correlation: Evaluated in shortcut audit; acceptable disparity bounds

## Splitting strategy

- Grouping key: `claim_id` and near-duplicate cluster ID
- Partition ratio: 70% Train (5,654 samples), 15% Validation (1,211 samples), 15% Test (1,214 samples)
- Random seed: `42`
- Manifest paths:
  - `data/manifests/fraud_train.csv` (5,654 rows)
  - `data/manifests/fraud_val.csv` (1,211 rows)
  - `data/manifests/fraud_test.csv` (1,214 rows)
- Leakage tests:
  - Zero intersection of `claim_id` groups across train, val, and test: PASSED
  - Zero intersection of SHA-256 hashes across train, val, and test: PASSED
  - Zero overlap of perceptual duplicate clusters across splits: PASSED
- Reason this split is safe: Prevents the model from memorizing specific vehicles, duplicate photos, or claim-specific lighting across train and test sets.

## Preprocessing

During audit:
- Grayscale conversion for Laplacian variance, brightness mean, and contrast std.
- Resizing to normalized dimensions only for perceptual hashing computation.
- Original image files remain strictly unmodified.

## Bias, limitations, and risk

1. **Shortcut Learning**: Image classification models may exploit background, camera device, watermark, or aspect-ratio discrepancies rather than genuine vehicle damage anomalies. The audit inspects feature distributions across classes.
2. **Provenance**: Web-scraped or public car damage datasets may not mirror actual insurance claims submitted to Indian insurers.
3. **Threshold Calibration**: The threshold must be tuned exclusively on validation data in Phase 2 to balance false positives against human investigator capacity.

## Privacy and security

- All raw images and extracted archives remain local and are excluded by `.gitignore`.
- EXIF metadata is inspected only for evidence integrity and not published.
- No personal identifying information (PII) is committed to Git.

## Validation decision

- Decision: **GO (with Responsible-AI Constraints)**
- Evidence: Dataset provides sufficient genuine and suspicious vehicle damage photographs to build a suspicious-image indicator; group-aware splitting eliminates split contamination.
- Required mitigation:
  1. The model output must always be reported as "Visual Fraud Risk", never "Confirmed Fraud".
  2. A high-risk score routes the claim to an insurance fraud investigator (`FRAUD_REVIEW`) and skips automated payment/cost approvals.
- Reviewer: Member 1 & Member 2
- Date: 2026-09-20
