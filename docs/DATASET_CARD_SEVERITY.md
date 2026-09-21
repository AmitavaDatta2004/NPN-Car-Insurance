# Dataset Card — Car Damage Severity Dataset (3 classes)

## Identity

- Dataset name: Car Damage Severity Dataset (Minor, Moderate, Severe)
- Internal version: v1.0-frozen
- Source URL: `kaggle:prajwalbhamere/car-damage-severity-dataset` (mirror of `anujms/car-damage-severity-dataset`)
- Download date: 2026-09-21
- Licence and permitted use: Public Research / Educational Use
- Original archive checksum: SHA-256 verified in dataset registry
- Maintainer: Member 3 (Severity ML A)
- Related tasks/notebooks: `SDATA-001` (`notebooks/05_severity_dataset_audit.ipynb`), Phase 5 (`06_severity_cnn_training.ipynb`), Phase 6 (`07_severity_mobilenetv2_training.ipynb`), Phase 7 (`08_severity_vit_tiny_training.ipynb`, `09_severity_model_comparison.ipynb`)

## Intended use

Supported task: Overall vehicle damage severity classification into three ordinal categories: `minor`, `moderate`, and `severe`.
Unsupported interpretations:
- Does NOT provide part-specific severity (e.g. door severity vs bumper severity). Part-specific severity requires localized annotations (Advanced Phase 20).
- Does NOT predict repair cost directly; cost is computed downstream by the rule-based Cost Engine (Phase 11).

## Structure and schema

| Field/path | Type | Meaning | Required | Notes |
| --- | --- | --- | --- | --- |
| `image_path` | string | Normalized repository-relative POSIX path | Yes | e.g. `data/raw/car_damage_severity/...` |
| `filename` | string | Base image file name | Yes | e.g. `0001.JPEG` |
| `label` | string | Categorical damage severity: `minor`, `moderate`, `severe` | Yes | Ordinal: minor < moderate < severe |
| `label_id` | integer | Numeric class index: 0, 1, 2 | Yes | Mapped via `severity_class_map.json` |
| `sha256` | string | SHA-256 64-character hexadecimal digest | Yes | Used for exact deduplication |
| `width` | integer | Image width in pixels | Yes | Decoded via OpenCV |
| `height` | integer | Image height in pixels | Yes | Decoded via OpenCV |
| `aspect_ratio` | float | Ratio of width to height | Yes | Rounded to 4 decimal places |
| `blur_score` | float | Laplacian variance blur metric | Yes | Higher values denote sharper focus |
| `brightness` | float | Mean grayscale pixel intensity | Yes | Range [0, 255] |
| `contrast` | float | Standard deviation of grayscale pixel intensity | Yes | Higher values denote greater contrast |
| `split` | string | Partition assignment: `train`, `val`, `test` | Yes | Frozen 70/15/15 partition |

## Label taxonomy

| Label | Operational definition | Count | Ambiguities |
| --- | --- | --- | --- |
| `minor` | Superficial scratches, shallow scuffs, small dings, minor paint scrapes with intact body panels | 534 (32.7%) | Boundary between deep scratches and small dent depressions |
| `moderate` | Large panel depressions, cracked plastic bumpers, dented doors/fenders requiring panel replacement or bodywork | 538 (33.0%) | Bumper displacement without chassis deformation |
| `severe` | Severe collision impacts, crushed engine compartments, rolled vehicles, deployed airbags, frame/chassis distortion | 559 (34.3%) | Rear-end collisions where hidden frame damage exists |

## Quality audit

- Total records/images: 1,631
- Decodable images: 1,631 (100.0%)
- Corrupt/missing: 0 (0.0%)
- Exact duplicates: 11 duplicate groups (22 total images) identified by SHA-256
- Near duplicates: 32 clusters (65 total images) identified by pHash (Hamming distance $\le 8$)
- Class balance: Well balanced (Minor: 32.7%, Moderate: 33.0%, Severe: 34.3%)
- Resolution and aspect-ratio distribution: Median dimensions 260×194; aspect ratios predominantly 1.33–1.50
- Watermark/source/background correlation: Low shortcut disparity (<0.15 relative difference across classes)
- Missing labels: None
- Annotation defects: Subjective boundary between upper minor and lower moderate damage

## Splitting strategy

- Grouping key: Clustered meta-ID merging exact duplicate SHA-256 groups and near-duplicate pHash clusters
- Train/validation/test counts:
  - Train: 1,140 images (69.9%) [Minor: 372, Moderate: 377, Severe: 391]
  - Val: 243 images (14.9%) [Minor: 80, Moderate: 80, Severe: 83]
  - Test: 248 images (15.2%) [Minor: 82, Moderate: 81, Severe: 85]
  - Total: 1,631 images (100.0%)
- Random seed: 42
- Manifest paths:
  - `data/manifests/severity_train.csv`
  - `data/manifests/severity_val.csv`
  - `data/manifests/severity_test.csv`
  - `data/manifests/severity_class_map.json`
  - `data/manifests/severity_manifest_summary.json`
  - `data/manifests/severity_audit_report.csv`
- Leakage tests: Programmatic assertions confirm zero intersection of file paths, SHA-256 hashes, or perceptual duplicate clusters across train, val, and test.
- Reason this split is safe: Duplicate pairs are constrained to the same partition, preventing identical or near-identical vehicles from appearing in both training and evaluation sets.

## Preprocessing

During Training:
- Resized to model input dimensions (e.g. 160×160 for baseline CNN, 224×224 for MobileNetV2 and ViT-Tiny)
- Normalization using ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`
- Gentle random horizontal flips and rotations (no aggressive filtering that destroys scratches)
During Evaluation & Runtime:
- Deterministic resize and normalization only

## Bias, limitations, and risk

- Geography: Mostly North American and European passenger vehicles; fewer commercial or auto-rickshaw vehicle types.
- Image-level annotation: Full image classified rather than specific parts.
- Lighting: Day-lit photographs dominate; night-time or rainy captures are underrepresented.

## Privacy and security

- No visible human faces in cropped damage areas.
- Number plates are either obscured, low resolution, or outside focal damage area.
- No personal policy or owner credentials contained in image pixels.

## Validation decision

- Decision: GO
- Evidence: 100% decodability (1,631/1,631 images), balanced class distribution, zero duplicate leakage across partitions, all 64 ML unit tests passing.
- Required mitigation: All three severity training phases (5, 6, 7) must strictly use these frozen manifests.
- Reviewer and date: Member 3 (Severity ML A) / Antigravity, 2026-09-21
