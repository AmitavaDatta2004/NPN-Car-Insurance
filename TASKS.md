# ClaimVision Task Board

Status values: `BACKLOG`, `READY`, `LOCKED`, `IN_PROGRESS`, `REVIEW`, `BLOCKED`, `DONE`.

## Task template

Copy this section for every task. A task may not enter `READY` without acceptance criteria.

```markdown
### TASK-ID ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Short title

- Phase:
- Owner:
- Reviewer:
- Status:
- Priority: P0 / P1 / P2
- Dependencies:
- Files allowed:
- Files prohibited:
- Objective:
- Inputs:
- Expected outputs:

Acceptance criteria:
- [ ] Criterion with an observable result
- [ ] Tests or notebook outputs specified
- [ ] Documentation updated

Validation commands:
- `command`

Risks/notes:
- Note
```

## Initial repository tasks

### CFG-001 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Install repository configuration pack

- Phase: 0
- Owner: Member 1
- Reviewer: Member 7
- Status: DONE
- Priority: P0
- Dependencies: Root `README.md` approved
- Files allowed: configuration files supplied in this pack
- Objective: establish one operating contract for all laptops and agents

Acceptance criteria:

- [x] Pack files exist at repository root using the documented paths.
- [x] `.env` is ignored and `.env.example` is committed.
- [x] All members confirm they read `AGENTS.md`.
- [x] Configuration commit is pushed to `main`.

Commit: `787c999` ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â feat: Add initial project configuration and CI/CD setup

---

### CFG-002 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Create application skeleton

- Phase: 0
- Owner: Member 1 (executed by Antigravity)
- Reviewer: Members 5 and 6
- Status: DONE
- Priority: P0
- Dependencies: CFG-001
- Files allowed: `backend/`, `frontend/`, `ml/`, `notebooks/`, `data/`, `artifacts/`, `scripts/`, `annotated/`, `reports/`, `demo/`, `docs/agent-work-log.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `config/project.yaml`, `.github/workflows/ci.yml`, `PROJECT_STATUS.md`, `TASKS.md`
- Files prohibited: `README.md`, `AGENTS.md`, `.env.example`, `.gitignore`, `.gitattributes`, `.editorconfig`, `.pre-commit-config.yaml`, all existing `docs/` templates, both `prompts/`
- Objective: create the repository structure defined in `README.md` Ãƒâ€šÃ‚Â§6 with working backend and frontend placeholders and an importable ML package
- Inputs: `README.md` Ãƒâ€šÃ‚Â§6 directory structure, `config/project.example.yaml`, `.github/workflows/ci.yml`

Acceptance criteria:

- [x] Full directory skeleton from README Ãƒâ€šÃ‚Â§6 exists with `.gitkeep` files where needed.
- [x] `claimvision_ml` Python package imports successfully from the venv (`import claimvision_ml`).
- [x] `claimvision_ml.__version__` returns `"0.1.0"`.
- [x] `pytest ml/tests/ -q` passes (package smoke test).
- [x] FastAPI health endpoint returns `{"status": "ok"}` at `GET /api/v1/health`.
- [x] `pytest backend/tests/ -q` passes (health smoke test).
- [x] `npm run build` succeeds inside `frontend/`.
- [x] All 16 notebook stubs exist in `notebooks/`.
- [x] CI workflow paths corrected to `backend/` and `frontend/`.
- [x] `config/project.yaml` committed (portable settings only).
- [x] `docs/agent-work-log.md` exists with Phase 0 entry.
- [x] `CHANGELOG.md` and `CONTRIBUTING.md` exist.
- [x] No dataset, model weight, secret, or local database is tracked.
- [x] `git status --short` is clean after all additions.

Commit: `abf99fb` ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Add initial implementation of claimvision_ml package with submodules for fraud detection, severity classification, and damage detection

Validation commands:

- `.venv\Scripts\python.exe -c "import claimvision_ml; print(claimvision_ml.__version__)"`
- `.venv\Scripts\pytest.exe ml/tests/ -q`
- `.venv\Scripts\pytest.exe backend/tests/ -q`
- `cd frontend && npm run build`
- `git status --short`
- `git diff --check`

Risks/notes:

- Heavy ML packages (torch, ultralytics) are listed in `ml/requirements.txt` but are NOT installed during Phase 0 to avoid multi-GB downloads. Each member installs them before Phase 1.
- `node_modules/` is not committed; each member runs `npm install` after pulling.

---

### DATA-001 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Validate fraud dataset feasibility

- Phase: 1
- Owner: Member 2 (executed by Antigravity)
- Reviewer: Member 1
- Status: DONE
- Priority: P0
- Dependencies: CFG-002
- Files allowed: `ml/src/claimvision_ml/data/`, `ml/src/claimvision_ml/quality/`, `notebooks/01_fraud_dataset_audit.ipynb`, `data/manifests/`, `docs/DATASET_CARD_FRAUD.md`, `ml/tests/test_data_audit.py`, `docs/agent-work-log.md`, `PROJECT_STATUS.md`, `TASKS.md`, `ml/requirements.txt`, `.gitignore`
- Files prohibited: other notebook files, backend routes, frontend app, severity/detection models
- Objective: determine whether the proposed labels support an honest suspicious-image classifier

Acceptance criteria:

- [x] Dataset source, licence, schema, counts, duplicates, corrupt images, and class balance documented.
- [x] Group-aware frozen manifests created with zero claim or duplicate leakage.
- [x] Source and watermark shortcut risks examined.
- [x] Representative images and failure cases displayed in notebook.
- [x] Written go/modify/stop recommendation recorded.
- [x] Unit tests for audit utilities and anti-leakage validation pass.

Evidence: 8,079 images audited; 0 corrupt files; 2,502 near-duplicate clusters grouped; zero-leakage 70/15/15 split (5,654 train, 1,211 val, 1,214 test) saved to `data/manifests/`; all 20 unit tests passing.

---

### ML-001 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Train and evaluate fraud baseline

- Phase: 2
- Owner: Member 2 / Antigravity
- Reviewer: Members 1 and 3
- Status: DONE
- Priority: P0
- Dependencies: DATA-001 accepted
- Files allowed: ml/src/claimvision_ml/fraud/, ml/tests/test_fraud_model.py,
  ml/artifacts/fraud/, ml/results/fraud/,
  notebooks/02_fraud_mobilenetv2_training.ipynb,
  notebooks/03_fraud_evaluation_and_threshold.ipynb,
  docs/MODEL_CARD_FRAUD_MNV2_V1.md, docs/EXPERIMENT_LOG.md,
  docs/agent-work-log.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md
- Files prohibited: all severity/detection/costing files, frontend, backend routes
- Objective: produce a calibrated suspicious-image signal before severity work begins

Acceptance criteria:

- [x] MobileNetV2 experiment is reproducible (seed=42, manifests frozen).
- [x] PR-AUC, class metrics, confusion matrix, calibration, and threshold trade-off shown (Val PR-AUC 0.4999, Test PR-AUC 0.5464, sweep 0.20Ã¢â‚¬â€œ0.80).
- [x] Untouched test set used exactly once for final reporting (held-out test set evaluated in Notebook 03 after threshold freeze).
- [x] Exported artifact passes smoke inference (`predict_fraud` verified standalone on sample image; outputs match).
- [x] Model card clearly limits the meaning of fraud output (`docs/MODEL_CARD_FRAUD_MNV2_V1.md`).

Evidence: MobileNetV2 fine-tuned with class weighting (FRAUD-MNV2-001); test PR-AUC=0.5464, test ROC-AUC=0.9077; high_threshold=0.80 gives 90.1% suspicious recall; ONNX exported and verified (max diff 8.94e-08); 11 plots in `ml/results/fraud/`; standalone `predict_fraud` verified; all 37 tests passing.

Validation commands:

- `.venv\Scripts\pytest.exe ml/tests/test_fraud_model.py -v`
- `.venv\Scripts\pytest.exe ml/tests/ -q`
- Run `notebooks/02_fraud_mobilenetv2_training.ipynb` top-to-bottom
- Run `notebooks/03_fraud_evaluation_and_threshold.ipynb` top-to-bottom

## Assignment rule

Only move one task per member into `IN_PROGRESS` at a time. Add the lock first, then update its status. When complete, link the commit and evidence under the task before marking `DONE`.

---

### CV-001 Ã¢â‚¬â€ OpenCV evidence-integrity runtime checker

- Phase: 3
- Owner: Member 3 (Severity ML A) / Antigravity
- Reviewer: Member 1
- Status: DONE
- Priority: P0
- Dependencies: ML-001 (fraud module), DATA-001 (quality module base)
- Files allowed: ml/src/claimvision_ml/quality/, ml/tests/test_quality_runtime.py, notebooks/04_opencv_quality_and_integrity.ipynb, docs/agent-work-log.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md
- Files prohibited: fraud/, severity/, detection/, backend routes, frontend app
- Objective: implement deterministic runtime evidence-integrity checks before any ML inference

Acceptance criteria:

- [x] `run_quality_checks()` returns `QualityResult` with route, reasons, and scores.
- [x] Corrupt/unreadable Ã¢â€ â€™ `MORE_EVIDENCE_REQUIRED`.
- [x] Resolution below 224Ãƒâ€”224 Ã¢â€ â€™ `MORE_EVIDENCE_REQUIRED`.
- [x] Blurry/dark/overexposed/low-contrast Ã¢â€ â€™ `rejection_reasons` populated.
- [x] Exact SHA-256 duplicate Ã¢â€ â€™ `DUPLICATE_REVIEW`.
- [x] Near dHash duplicate (Hamming Ã¢â€°Â¤ 4) Ã¢â€ â€™ `DUPLICATE_REVIEW`.
- [x] EXIF absent Ã¢â€ â€™ `warnings["exif_absent"]` only; never sets `passed=False` or `route=FRAUD_REVIEW`.
- [x] `draw_bounding_boxes()` and `save_annotated_image()` work and preserve original.
- [x] 19 unit tests pass; full suite (56 tests) passes with no regression.
- [x] Notebook 04 has 12 sections using real Colab dataset images.
- [x] `ruff check ml/` Ã¢â‚¬â€ only 1 pre-existing Phase 2 warning remains (not our files).

Evidence: 19 new tests pass (56 total); `quality/runtime_checker.py` and extended `image_checks.py` committed; notebook 04 fully implemented with real dataset path configuration.

Validation commands:

- `.venv\Scripts\pytest.exe ml/tests/test_quality_runtime.py -v`
- `.venv\Scripts\pytest.exe ml/tests/ -q`
- Run `notebooks/04_opencv_quality_and_integrity.ipynb` top-to-bottom in Colab

---

### SDATA-001 â€” Severity dataset audit and manifest freeze

- Phase: 4
- Owner: Member 3 (Severity ML A) / Antigravity
- Reviewer: Member 1
- Status: DONE
- Priority: P0
- Dependencies: CFG-002, CV-001
- Files allowed: ml/src/claimvision_ml/data/severity_audit.py, ml/src/claimvision_ml/data/__init__.py, ml/tests/test_severity_audit.py, ml/tests/conftest.py, ml/pyproject.toml, notebooks/05_severity_dataset_audit.ipynb, data/manifests/severity_*, docs/DATASET_CARD_SEVERITY.md, ml/results/severity/, docs/agent-work-log.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md
- Files prohibited: fraud/, backend routes, frontend app, detection models
- Objective: audit the 3-class Car Damage Severity Dataset (1,631 images), verify image decodability, detect exact and perceptual duplicates, analyze shortcut risks, and freeze duplicate-safe 70/15/15 stratified manifests for CNN, MobileNetV2, and ViT-Tiny

Acceptance criteria:
- [x] All 1,631 images validated for OpenCV decode integrity with zero corrupt images unquarantined (100.0% decodable).
- [x] Exact SHA-256 duplicate groups identified and cataloged (11 groups, 22 total images).
- [x] Perceptual hash near-duplicate clusters (pHash dist <= 8) identified (32 clusters, 65 total images).
- [x] Class distributions (Minor: 534, Moderate: 538, Severe: 559) and quality distributions (dimensions, blur, brightness, contrast) analyzed across classes.
- [x] Duplicate-safe 70/15/15 stratified train, validation, and test manifests generated (Train: 1,140, Val: 243, Test: 248).
- [x] Zero-leakage verification: programmatic assertions confirm 0 SHA-256 or duplicate cluster leakage across splits.
- [x] Unit tests for severity audit utilities pass without regression (8/8 new pass; all 64 ML suite tests pass).
- [x] Notebook 05 executed with clean, curated judge-facing outputs and Colab/local universal sync.
- [x] Dataset card `docs/DATASET_CARD_SEVERITY.md` created.
- [x] Manifest summary and class map exported to `data/manifests/`.

Evidence: 1,631 images audited with 0 corrupt; 11 exact duplicate groups & 32 pHash clusters grouped; zero-leakage 70/15/15 split (1,140 train, 243 val, 248 test) frozen in `data/manifests/`; all 64 ML tests pass; `docs/DATASET_CARD_SEVERITY.md` and 4 visual charts committed.

Validation commands:
- `.venv\Scripts\pytest.exe ml/tests/test_severity_audit.py -v`
- `.venv\Scripts\pytest.exe ml/tests/ -q`
- Run `notebooks/05_severity_dataset_audit.ipynb` top-to-bottom

---

### SEV-CNN-001 — Severity Baseline CNN Training (Phase 5)

- Phase: 5
- Owner: Member 3 / Antigravity
- Reviewer: Member 1
- Status: DONE
- Priority: P0
- Dependencies: SDATA-001 (DONE — manifests frozen)
- Files allowed: ml/src/claimvision_ml/severity/cnn.py, ml/src/claimvision_ml/severity/__init__.py, ml/tests/test_severity_cnn.py, ml/results/severity/cnn_metrics.json, notebooks/06_severity_cnn_training.ipynb, docs/EXPERIMENT_LOG.md, docs/agent-work-log.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md
- Files prohibited: notebooks/07_severity_mobilenetv2_training.ipynb, notebooks/08_severity_vit_tiny_training.ipynb, ml/src/claimvision_ml/severity/mobilenet.py, ml/src/claimvision_ml/severity/vit.py
- Objective: implement and train a 4-block convolutional neural network from scratch on the frozen Phase 4 severity manifests (160×160 input, SEED=42) establishing the baseline performance floor for Phase 6 and 7.

Acceptance criteria:
- [x] `ml/src/claimvision_ml/severity/cnn.py` created with SeverityCNN, SeverityDataset, get_severity_transforms, build_cnn_model, load_cnn_model, predict_severity_cnn, measure_cpu_latency, export_onnx.
- [x] `notebooks/06_severity_cnn_training.ipynb` complete with 17 cells covering all README §10 required outputs.
- [x] Notebook runs bimodal (Colab GPU + local CPU) with self-healing fallback.
- [x] Test set evaluated EXACTLY ONCE in the final dedicated cell (accuracy 0.5968, macro F1 0.5921).
- [x] `ml/tests/test_severity_cnn.py` created with 29 unit tests.
- [x] All existing ML tests continue to pass without regression.
- [x] `ml/results/severity/cnn_metrics.json` committed with real verified metrics.
- [x] `docs/EXPERIMENT_LOG.md` updated with SEV-CNN-001 registry entry and detailed report.
- [x] Phase 5 gate passed: checkpoint loads outside notebook; curves, metrics, confusion matrix, and error examples visible.

Evidence: SeverityCNN trained from scratch in 19 epochs; Best Val Macro F1: 0.6206; Test Macro F1: 0.5921; Test Accuracy: 59.68%; CPU latency: 17.90 ms/image; ONNX exported; 29 unit tests pass; notebook 06 executed with all outputs in Colab.

Validation commands:
- `.venv\Scripts\pytest.exe ml/tests/test_severity_cnn.py -v`
- `.venv\Scripts\pytest.exe ml/tests/ -q`
- Run `notebooks/06_severity_cnn_training.ipynb` top-to-bottom

---

### ML-003 — Per-Epoch Balanced Resampling Comparison (50:50, 40:60, 30:70, 20:80)

- Phase: 2 (Extension)
- Owner: Member 2 / Antigravity
- Reviewer: Member 1
- Status: DONE
- Priority: P1
- Dependencies: ML-001 accepted
- Files allowed: ml/src/claimvision_ml/fraud/, ml/tests/test_balanced_sampler.py,
  ml/artifacts/fraud/balanced/, ml/results/fraud/balanced_*,
  notebooks/02b_fraud_balanced_resampling_comparison.ipynb,
  docs/MODEL_CARD_FRAUD_MNV2_V1.md, docs/EXPERIMENT_LOG.md,
  docs/agent-work-log.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md
- Files prohibited: severity/, detection/, costing/, backend routes, frontend app
- Objective: train 4 MobileNetV2 models with per-epoch balanced resampling (50:50, 40:60, 30:70, 20:80) keeping suspicious class fixed (325 train) and rotating genuine class; compare on held-out test set against baseline

Acceptance criteria:
- [x] `BalancedEpochSampler` implemented with fixed suspicious class and rotating genuine sample per epoch.
- [x] 12 unit tests pass in `test_balanced_sampler.py`; full suite passes with no regression.
- [x] Comparison notebook `02b_fraud_balanced_resampling_comparison.ipynb` executed top-to-bottom.
- [x] Side-by-side comparison table, PR curves, and confusion matrices generated across all 4 ratios + baseline.
- [x] Best model checkpoint and threshold JSON exported (`20:80` won with Test PR-AUC 0.5617, Recall 66.2%, F1 0.5000).
- [x] Experiment log and model card updated with real metrics.

Evidence: All 4 models trained on GPU (CUDA); 20:80 selected as winner (Test PR-AUC 0.5617 vs Baseline 0.5464, FP dropped from 393 to 70, F1 doubled to 0.5000); all 5 comparison figures extracted to `ml/results/fraud/`; `comparison_results.json` and `best_model_thresholds.json` committed.

Validation commands:
- `.venv\Scripts\pytest.exe ml/tests/test_balanced_sampler.py -v`
- `.venv\Scripts\pytest.exe ml/tests/ -q`
- Run `notebooks/02b_fraud_balanced_resampling_comparison.ipynb` top-to-bottom

---

### SEV-MNV2-001 — Severity MobileNetV2 transfer learning

- Phase: 6
- Owner: Friend 2 / Antigravity
- Reviewer: Member 1 / Member 3
- Status: DONE
- Priority: P0
- Dependencies: SDATA-001 accepted
- Files allowed: ml/src/claimvision_ml/severity/mobilenet.py, ml/src/claimvision_ml/severity/dataset.py, ml/src/claimvision_ml/severity/predict.py, ml/src/claimvision_ml/severity/__init__.py, ml/tests/test_severity_mobilenet.py, ml/artifacts/severity/, ml/results/severity/, ml/results/severity_mnv2_metrics.json, notebooks/07_severity_mobilenetv2_training.ipynb, docs/MODEL_CARD_SEVERITY_MNV2_V1.md, docs/EXPERIMENT_LOG.md, docs/agent-work-log.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md
- Files prohibited: notebooks/06_severity_cnn_training.ipynb, ml/src/claimvision_ml/severity/cnn.py, notebooks/08_severity_vit_tiny_training.ipynb, ml/src/claimvision_ml/severity/vit.py, backend routes, frontend app, detection models
- Objective: train, validate, evaluate, and export a lightweight two-stage MobileNetV2 transfer-learning severity classifier (minor, moderate, severe) on the frozen 70/15/15 manifests.

Acceptance criteria:
- [x] PyTorch Dataset reads frozen manifests (`severity_train.csv`, `severity_val.csv`, `severity_test.csv`) with bimodal path resolution (local & Colab).
- [x] MobileNetV2 model architecture with ImageNet weights, pooling, dropout (0.3), and 3-class linear head implemented.
- [x] Two-stage transfer learning: Stage A (frozen backbone, LR=1e-3, head only) and Stage B (unfrozen top blocks features[17:], LR=1e-5).
- [x] Checkpoint best model on validation Macro F1 score (no test set leakage during training/tuning).
- [x] Evaluated strictly once on untouched held-out test manifest (`severity_test.csv`).
- [x] Metrics reported: Accuracy, Macro Precision/Recall/F1, Weighted F1, Severe Recall, confusion matrix, CPU/GPU latency.
- [x] Model weights exported (`.pt` and `.onnx` format); ONNX output parity verified within 1e-4.
- [x] Standalone `predict_severity` runtime function implemented and verified.
- [x] Unit tests pass in `ml/tests/test_severity_mobilenet.py` and full suite passes without regression (73 passed, 1 skipped).
- [x] Judge-facing notebook `07_severity_mobilenetv2_training.ipynb` contains clean, curated, reproducible outputs and bimodal Colab/local support.
- [x] Model card `docs/MODEL_CARD_SEVERITY_MNV2_V1.md` and `docs/EXPERIMENT_LOG.md` entry created.

Evidence: MobileNetV2 two-stage transfer learning module (`mobilenet.py`), dataset loader (`dataset.py`), standalone inference engine (`predict.py`), and 20-cell bimodal judge notebook 07 created; all 10 severity tests pass; full suite 74 tests green (73 passed, 1 skipped); model card `docs/MODEL_CARD_SEVERITY_MNV2_V1.md` committed.

Validation commands:
- `.venv\Scripts\pytest.exe ml/tests/test_severity_mobilenet.py -v`
- `.venv\Scripts\pytest.exe ml/tests/ -q`
- Run `notebooks/07_severity_mobilenetv2_training.ipynb` top-to-bottom

---

### SEV-VIT-001 — Train and evaluate ViT-Tiny severity classifier

- Phase: 7
- Owner: Member 4 / Antigravity
- Reviewer: Members 1 and 3
- Status: DONE
- Priority: P0
- Dependencies: SDATA-001 accepted
- Files allowed: `ml/src/claimvision_ml/severity/vit.py`, `ml/src/claimvision_ml/severity/__init__.py`, `ml/tests/test_severity_vit.py`, `notebooks/08_severity_vit_tiny_training.ipynb`, `scripts/build_notebook_08.py`, `scripts/run_severity_vit.py`, `docs/MODEL_CARD_SEVERITY_VIT_TINY.md`, `ml/results/severity/vit/`, `ml/artifacts/severity/vit/`, `docs/EXPERIMENT_LOG.md`, `docs/agent-work-log.md`, `PROJECT_STATUS.md`, `TASKS.md`, `TASK_LOCKS.md`
- Files prohibited: `notebooks/06_severity_cnn_training.ipynb`, `notebooks/07_severity_mobilenetv2_training.ipynb`, `notebooks/09_severity_model_comparison.ipynb`, other severity model files, backend routes, frontend app, detection models, fraud models
- Objective: train, evaluate, and export the ViT-Tiny (`vit_tiny_patch16_224`) transformer model on the frozen Car Damage Severity manifests (1,140 train, 243 val, 248 test)

Acceptance criteria:
- [x] ViT-Tiny architecture implemented using `vit_tiny_patch16_224` with 3 output classes (`minor`, `moderate`, `severe`).
- [x] Two-stage training protocol: Stage A head warmup (frozen backbone) + Stage B progressive fine-tuning with cosine decay.
- [x] Evaluated on validation set for early stopping using macro F1 (best Val Macro F1: 0.1697).
- [x] Held-out test set (`severity_test.csv`, 248 images) evaluated strictly once.
- [x] Multiclass confusion matrix, per-class metrics, severe recall (100%), accuracy (34.27%), and macro F1 (0.1702) reported.
- [x] ONNX export verified and output discrepancy checked.
- [x] Model card `docs/MODEL_CARD_SEVERITY_VIT_TINY.md` created.
- [x] Notebook 08 implemented with all 18 required sections and judge-facing outputs.
- [x] Unit tests for ViT module pass (8/8 pass); full test suite passes without regressions (72/72 tests pass).

Evidence: 1,631 images evaluated across frozen 70/15/15 split; 5.52M param ViT-Tiny trained in 2 stages; CPU latency 12.60 ms/image; ONNX exported to `artifacts/models/severity_vit.onnx`; 72 unit tests passing; plots saved in `ml/results/severity/vit/`.

Validation commands:
- `.venv\Scripts\pytest.exe ml/tests/test_severity_vit.py -v`
- `.venv\Scripts\pytest.exe ml/tests/ -q`
- Run `notebooks/08_severity_vit_tiny_training.ipynb` top-to-bottom

---

### DET-COCO-001 — COCO annotation audit and YOLO conversion

- Phase: 8
- Owner: Member 4 (Detection owner)
- Reviewer: Member 1
- Status: DONE
- Priority: P0
- Dependencies: SEV-VIT-001 DONE (Phase 7 complete)
- Files allowed: ml/src/claimvision_ml/detection/, ml/tests/test_coco_converter.py, ml/results/detection/, notebooks/10_coco_annotation_audit_and_conversion.ipynb, docs/EXPERIMENT_LOG.md, docs/agent-work-log.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md
- Files prohibited: fraud/, severity/, backend routes, frontend app, notebooks 06, 07, 08, 09, 11, 12, 13, 14, 15
- Objective: Audit the COCO car-damage detection dataset (59 train / 11 val / 8 test images, generic damage + 5 part classes at /content/NPN Car Insurance/data/raw/ on Colab), verify bounding boxes visually, convert to YOLO format, and prove correctness with assertions before YOLO training begins in Phase 9.
- Inputs:
  - train/: 59 images, COCO_train_annos.json, COCO_mul_train_annos.json
  - val/: 11 images, COCO_val_annos.json, COCO_mul_val_annos.json
  - test/: 8 images (unannotated test split)

Acceptance criteria:
- [x] `coco_converter.py` created with `load_coco_json`, `validate_structure`, `draw_coco_boxes`, `convert_bbox_to_yolo`, `convert_split`, `write_data_yaml`, `run_conversion_assertions`.
- [x] Round-trip conversion assertion passes within 1e-6 tolerance.
- [x] 17 unit tests pass in `test_coco_converter.py`; full ML suite (108 tests) passes without regression.
- [x] Notebook 10 implemented with 14 cells: category counts, before/after box grids, assertion output, data.yaml printed, directory tree.
- [x] `run_conversion_assertions` prints PASS for both `yolo_damage/` and `yolo_parts/` across all 3 splits.
- [x] `data.yaml` for damage: `nc=1`, `names=[damage]`.
- [x] `data.yaml` for parts: `nc=5`, `names=[headlamp, front_bumper, hood, door, rear_bumper]`.
- [x] Every image has a `.txt` label file (empty file for unannotated test split).
- [x] No YOLO training in Notebook 10.
- [x] Notebook 10 executed with all visible outputs preserved.
- [x] `EXPERIMENT_LOG.md` updated with `DET-COCO-001` entry.
- [x] `PROJECT_STATUS.md` Phase 8 → Complete.

Evidence: All 78 images across train (59), val (11), and test (8) successfully converted into `yolo_damage/` and `yolo_parts/`; conversion assertions PASS across all splits; 17/17 unit tests pass; Notebook 10 executed top-to-bottom with full visible outputs.

Validation commands:
- `.venv\Scripts\pytest.exe ml/tests/test_coco_converter.py -v`
- `.venv\Scripts\pytest.exe ml/tests/ -q`
- Run `notebooks/10_coco_annotation_audit_and_conversion.ipynb` top-to-bottom in Colab

Risks/notes:
- Dataset is very small (59 train images). Document the generalisation limitation prominently.
- Some images may have no annotations — an empty .txt file must still be created for them.
- Part class names in the COCO JSON may differ slightly from the YOLO names (e.g. "rear bumper" vs "rear_bumper") — the converter must handle the mapping explicitly.

---

### DET-YOLO-001 — Generic Damage YOLOv8 Training

- Phase: 9
- Owner: Member 4 (Detection ML)
- Reviewer: Member 1 / Member 5
- Status: DONE
- Priority: P0
- Dependencies: DET-COCO-001 DONE
- Files allowed: ml/src/claimvision_ml/detection/, ml/tests/test_damage_detector.py, notebooks/11_yolo_damage_training.ipynb, docs/MODEL_CARD_DAMAGE_YOLO_V1.md, docs/EXPERIMENT_LOG.md, docs/agent-work-log.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md
- Files prohibited: fraud/, severity/, costing/, backend routes, frontend app, notebooks 06, 07, 08, 09, 10, 12, 13, 14, 15
- Objective: train, validate, and export a lightweight YOLOv8n single-class generic damage detector (nc=1, name='damage') on the converted COCO dataset from Phase 8. Provide reusable inference module, visual bounding box overlay generator, export PyTorch and ONNX models, and execute reproducible judge notebook 11.
- Inputs:
  - `ml/results/detection/yolo_damage/data.yaml`
  - `train/`: 59 images + labels
  - `val/`: 11 images + labels
  - `test/`: 8 images (held-out test split)

Acceptance criteria:
- [x] `ml/src/claimvision_ml/detection/damage.py` implemented with `DamageDetection`, `DamageDetector`, `detect_damage`, `export_damage_onnx`.
- [x] Public API exposed in `ml/src/claimvision_ml/detection/__init__.py`.
- [x] Visual bounding box overlay drawn cleanly on an image copy with class label and confidence percentage; original image remains pristine.
- [x] No-detection edge case handled gracefully (empty list returned, no exception).
- [x] Unit tests in `ml/tests/test_damage_detector.py` pass (17 tests); full test suite passes with 0 regressions (125 passed).
- [x] Notebook 11 (`11_yolo_damage_training.ipynb`) implemented with 17 cells adhering to README §10 and AGENTS.md §7.
- [x] Bimodal execution support (Google Colab GPU / Local CPU) with dataset auto-sync fallback.
- [x] Held-out test set evaluated strictly ONCE.
- [x] Model checkpoints exported: `artifacts/models/damage_yolov8n.pt` and `artifacts/models/damage_yolov8n.onnx`.
- [x] `docs/MODEL_CARD_DAMAGE_YOLO_V1.md` and `docs/EXPERIMENT_LOG.md` entry created.
- [x] `PROJECT_STATUS.md` and `TASK_LOCKS.md` updated.

Evidence: Reusable `DamageDetector` engine, coordinate converters, and overlay visualizer created in `damage.py`; all 17 detector unit tests pass; full test suite (125 tests) passes green without regressions; 17-cell judge-ready Notebook 11 generated with bimodal Colab/local support; Model card `MODEL_CARD_DAMAGE_YOLO_V1.md` committed.

Validation commands:
- `.venv\Scripts\pytest.exe ml/tests/test_damage_detector.py -v`
- `.venv\Scripts\pytest.exe ml/tests/ -q`
- Run `notebooks/11_yolo_damage_training.ipynb` top-to-bottom

---

### DET-PART-001 — Damaged-Part YOLOv8 Training (5 Classes)

- Phase: 10
- Owner: Member 4 (Detection ML)
- Reviewer: Member 1 / Member 5
- Status: DONE
- Priority: P0
- Dependencies: DET-YOLO-001 DONE
- Files allowed: ml/src/claimvision_ml/detection/, ml/tests/test_part_detector.py, notebooks/12_yolo_part_training.ipynb, docs/MODEL_CARD_PART_YOLO_V1.md, docs/EXPERIMENT_LOG.md, docs/agent-work-log.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md
- Files prohibited: fraud/, severity/, costing/, backend routes, frontend app, notebooks 06, 07, 08, 09, 10, 11, 13, 14, 15
- Objective: train, validate, and evaluate a lightweight YOLOv8n 5-class damaged vehicle part detector (nc=5: headlamp, front_bumper, hood, door, rear_bumper) on the converted COCO dataset from Phase 8. Provide reusable inference module, multi-color component visual overlays, export PyTorch and ONNX models, and execute reproducible judge notebook 12 with an explicit Go/No-Go decision gate for production demo.
- Inputs:
  - `ml/results/detection/yolo_parts/data.yaml`
  - `train/`: 59 images + labels (177 part boxes)
  - `val/`: 11 images + labels
  - `test/`: 8 images (held-out test split)

Acceptance criteria:
- [x] `ml/src/claimvision_ml/detection/parts.py` implemented with `PartDetection`, `PartDetector`, `detect_parts`, `export_parts_onnx`, and `get_detected_part_names`.
- [x] Public API exposed in `ml/src/claimvision_ml/detection/__init__.py`.
- [x] Multi-color visual bounding box overlay drawn cleanly with distinct colors per vehicle part and confidence percentages; original image remains pristine.
- [x] No-detection edge case handled gracefully (empty list returned, no exception).
- [x] Unit tests in `ml/tests/test_part_detector.py` pass; full test suite passes with 0 regressions.
- [x] Notebook 12 (`12_yolo_part_training.ipynb`) implemented with 18 cells adhering to README §10 and AGENTS.md §7.
- [x] Per-class performance reviewed (Precision, Recall, AP per class).
- [x] Front-vs-rear bumper confusion analyzed.
- [x] Bimodal execution support (Google Colab GPU / Local CPU) with dataset auto-sync fallback.
- [x] Held-out test set evaluated strictly ONCE.
- [x] Explicit production-demo vs. experimental Go/No-Go decision recorded (NO-GO for YOLO part detection; Decoupled Damage YOLO + Location CNN architecture selected).
- [x] Model checkpoints exported: `artifacts/models/parts_yolov8n.pt` and `artifacts/models/parts_yolov8n.onnx`.
- [x] `docs/MODEL_CARD_PART_YOLO_V1.md` and `docs/EXPERIMENT_LOG.md` entry created.
- [x] `PROJECT_STATUS.md` updated.

Evidence: Option A selected. Bimodal YOLO part detection module and tests pass (54 detection tests green, 252 full suite pass); Notebook 12 executed and documented with explicit NO-GO Gate Decision (59 train / 1 val image insufficient for 5-class bounding box regression; production uses decoupled Damage YOLO + Location CNN architecture). Notebook 12 serves as academic ablation study for judges.

Validation commands:
- `.venv\Scripts\pytest.exe ml/tests/test_part_detector.py -v`
- `.venv\Scripts\pytest.exe ml/tests/ -q`
- Run `notebooks/12_yolo_part_training.ipynb` top-to-bottom

---

### LOC-DATA-001 — Location Dataset Module & Label Derivation

- Phase: 10b
- Owner: Location CNN member / Antigravity
- Reviewer: Member 1 / Member 4
- Status: DONE
- Priority: P0
- Dependencies: DET-COCO-001 DONE (COCO annotations available)
- Files allowed: ml/src/claimvision_ml/location/, ml/tests/test_location_dataset.py, ml/tests/test_location_classifier.py, docs/DATASET_CARD_LOCATION.md, docs/EXPERIMENT_LOG.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md, README.md
- Files prohibited: fraud/, severity/, detection/, backend routes, frontend app, notebooks 06–12
- Objective: implement the `claimvision_ml.location` Python subpackage with `LocationDataset`, `derive_location_labels`, `get_location_transforms`, `LocationClassification`, `LocationClassifier`, `classify_location`, and `export_location_onnx`. Write unit tests for all public API.
- Inputs: COCO Car Damage annotations (COCO_mul_train_annos.json, COCO_mul_val_annos.json); 5 part classes from Phase 8 conversion.

Acceptance criteria:
- [x] `ml/src/claimvision_ml/location/__init__.py` exports all public API.
- [x] `derive_location_labels()` correctly assigns dominant-part label (count-based, area tiebreak).
- [x] `derive_location_labels()` skips images with no recognised-class annotations.
- [x] `derive_location_labels()` normalises category names (`"front bumper"` → `"front_bumper"`).
- [x] `load_location_splits()` handles sparse validation folders via stratified pooling fallback.
- [x] `LocationDataset.__getitem__()` returns tensor shape (3, 224, 224) and torch.long label.
- [x] `LocationDataset.class_weights()` returns tensor of shape (5,) with normalized mean = 1.0.
- [x] `LocationClassification` raises ValueError for invalid class_id, confidence, or class_name.
- [x] `classify_location()` returns correct class and emits warning when confidence < 0.40.
- [x] `export_location_onnx()` skipped when onnxscript absent (matches existing pattern).
- [x] 43 passed, 1 skipped in `test_location_dataset.py` + `test_location_classifier.py`.
- [x] Full test suite (188 passed, 5 skipped) — zero regressions.
- [x] `docs/DATASET_CARD_LOCATION.md` written.
- [x] `README.md` §6, §7.2, §10, §11 updated.
- [x] `docs/DECISIONS.md` ADR-004 written.
- [x] `docs/EXPERIMENT_LOG.md` LOC registry rows added.

Evidence: All 188 tests pass. `from claimvision_ml.location import LocationClassifier, load_location_splits` imports cleanly.

Validation commands:
- `.venv\Scripts\python.exe -m pytest ml/tests/test_location_dataset.py ml/tests/test_location_classifier.py -v`
- `.venv\Scripts\python.exe -m pytest ml/tests/ -q`
- `.venv\Scripts\python.exe -c "from claimvision_ml.location import LocationClassifier, load_location_splits; print('OK')"`

---

### LOC-MNV2-001 — Location MobileNetV2 Classifier

- Phase: 11a
- Owner: Location CNN member / Antigravity
- Reviewer: Member 1 / Member 4
- Status: DONE
- Priority: P0
- Dependencies: LOC-DATA-001 DONE
- Files allowed: ml/src/claimvision_ml/location/, ml/tests/test_location_classifier.py, notebooks/13_location_mobilenetv2_training.ipynb, scripts/build_notebook_13.py, docs/MODEL_CARD_LOCATION_MNV2_V1.md, docs/EXPERIMENT_LOG.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md
- Files prohibited: fraud/, severity/, detection/ (except reading), backend routes, frontend app, notebooks 06–12, 14, 15
- Objective: implement and verify the MobileNetV2 image-level location classifier, 2-stage fine-tuning architecture (head training + block unfreezing), inference runtime, unit tests, and judge-facing training notebook 13.

Acceptance criteria:
- [x] `LocationMobileNet` class implemented with 2-stage fine-tuning head in `ml/src/claimvision_ml/location/mobilenet.py`.
- [x] `freeze_backbone()`, `unfreeze_last_blocks()`, and `measure_cpu_latency()` implemented and tested.
- [x] Notebook 13 generated via `scripts/build_notebook_13.py` with 18 structured cells.
- [x] Colab environment setup configured without Google Drive mount dependency.
- [x] Stratified 80/20 train/val split (48 train / 12 val) integrated to resolve 1-image `val/` folder defect.
- [x] Classification report and confusion matrix updated with `labels=list(range(5))` and `zero_division=0`.
- [x] Unannotated held-out test set evaluated via forward inference mode with predictions and top-2 alternatives.
- [x] Unit tests in `test_location_classifier.py` passing (freeze, unfreeze, latency, forward shapes).
- [x] `docs/MODEL_CARD_LOCATION_MNV2_V1.md` created.
- [x] `PROJECT_STATUS.md` Phase 11a updated.

Validation commands:
- `.venv\Scripts\python.exe -m pytest ml/tests/test_location_classifier.py -v`
- `.venv\Scripts\python.exe scripts/build_notebook_13.py`

---

### LOC-EFF-001 — Location EfficientNet-B0 Classifier

- Phase: 11b
- Owner: Location CNN member / Antigravity
- Reviewer: Member 1 / Member 4
- Status: DONE
- Priority: P0
- Dependencies: LOC-MNV2-001 DONE (same data pipeline verified)
- Files allowed: ml/src/claimvision_ml/location/, notebooks/14_location_efficientnet_training.ipynb, scripts/build_notebook_14.py, docs/MODEL_CARD_LOCATION_EFF_V1.md, docs/EXPERIMENT_LOG.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md
- Files prohibited: fraud/, severity/, detection/ (except reading), backend routes, frontend app, notebooks 06–13, 15
- Objective: implement and verify the EfficientNet-B0 image-level location classifier (via `timm`), 2-stage fine-tuning architecture, inference runtime, unit tests, and judge-facing training notebook 14.

Acceptance criteria:
- [x] `LocationEfficientNet` implemented via `timm` with custom classification head in `ml/src/claimvision_ml/location/efficientnet.py`.
- [x] `freeze_backbone()`, `unfreeze_last_blocks()`, and `measure_cpu_latency()` implemented and tested.
- [x] Notebook 14 generated via `scripts/build_notebook_14.py` with identical evaluation structure to Notebook 13.
- [x] Colab environment setup configured without Google Drive mount dependency.
- [x] Stratified 80/20 train/val split (48 train / 12 val) integrated to resolve 1-image `val/` folder defect.
- [x] Classification report and confusion matrix updated with `labels=list(range(5))` and `zero_division=0`.
- [x] Unannotated held-out test set evaluated via forward inference mode with predictions and top-2 alternatives.
- [x] Unit tests in `test_location_classifier.py` passing for EfficientNet-B0 model type.
- [x] `docs/MODEL_CARD_LOCATION_EFF_V1.md` created.
- [x] `PROJECT_STATUS.md` Phase 11b updated.

Validation commands:
- `.venv\Scripts\python.exe -m pytest ml/tests/test_location_classifier.py -v`
- `.venv\Scripts\python.exe scripts/build_notebook_14.py`

---

### LOC-COMP-001 — Location Model Comparison

- Phase: 11c
- Owner: Location CNN member
- Reviewer: Member 1 / Member 4 / Member 7 (presentation)
- Status: READY
- Priority: P1
- Dependencies: LOC-MNV2-001 DONE, LOC-EFF-001 DONE
- Files allowed: notebooks/15_location_model_comparison.ipynb, scripts/build_notebook_15.py, docs/EXPERIMENT_LOG.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md
- Files prohibited: All other notebooks and modules
- Objective: generate Notebook 15 to compare both models side-by-side on identical validation splits, fill in the selection decision table, and document the selected winner.

Acceptance criteria:
- [x] Notebook 15 generated via `scripts/build_notebook_15.py` with 12 structured cells.
- [x] Model loader imports both `LocationMobileNet` and `LocationEfficientNet`.
- [x] Evaluation function updated with `labels=list(range(5))` and `zero_division=0`.
- [x] Side-by-side metric comparison, confusion matrix heatmaps, and per-class F1 bar charts scaffolded.
- [x] CPU latency and model size benchmark comparison scaffolded.
- [ ] Run Notebook 15 top-to-bottom in Colab with saved checkpoints from NB 13 & 14.
- [ ] Selection decision table completed (winner, reason, experiment ID).
- [ ] `docs/EXPERIMENT_LOG.md` LOC-COMP-001 updated with final decision.
- [ ] `PROJECT_STATUS.md` Phase 11c updated to Complete.

Validation commands:
- `.venv\Scripts\python.exe scripts/build_notebook_15.py`

---

### INF-001 — Unified Inference Pipeline & Orchestrator

- Phase: 12 (Unified Inference)
- Owner: Member 5 (Backend/ML) / Antigravity
- Reviewer: Member 1
- Status: DONE
- Priority: P0
- Dependencies: Quality checks (CV-001), Fraud model (ML-001), Severity model (SEV-MNV2-001), Detection model (DET-YOLO-001), Location model (LOC-DATA-001)
- Files allowed: ml/src/claimvision_ml/pipeline/, ml/src/claimvision_ml/costing/, config/cost_table.json, config/decision_thresholds.yaml, ml/tests/test_pipeline_assess.py, ml/tests/test_decision.py, ml/tests/test_cost_engine.py, notebooks/16_unified_inference_demo.ipynb, docs/agent-work-log.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md
- Files prohibited: fraud/, severity/, detection/, location/ (except importing/reading), backend/, frontend/
- Objective: implement the unified `assess_claim()` pipeline in `claimvision_ml.pipeline` chaining quality -> fraud -> severity -> detection -> location -> costing -> decision routing. Support mock/heuristic fallback when weights are not locally present. Provide cost engine and configurable decision thresholds. Write comprehensive unit tests.

Acceptance criteria:
- [x] `config/cost_table.json` created with illustrative repair/replacement costs per part and severity tier (in INR).
- [x] `config/decision_thresholds.yaml` created with named threshold configurations for blur, fraud risk, cost limits, and confidence.
- [x] `ml/src/claimvision_ml/costing/` implements `estimate_cost()` returning `CostEstimate` dataclass.
- [x] `ml/src/claimvision_ml/pipeline/schemas.py` defines `AssessmentResult` and nested dataclasses matching API contracts.
- [x] `ml/src/claimvision_ml/pipeline/decision.py` implements routing logic returning appropriate routes and reason codes.
- [x] `ml/src/claimvision_ml/pipeline/assess.py` implements `assess_claim()` with graceful real/mock execution.
- [x] High fraud risk stops automated damage and costing evaluation and routes to `FRAUD_REVIEW`.
- [x] Unit tests for cost engine, decision routing, and full pipeline pass (`test_cost_engine.py`, `test_decision.py`, `test_pipeline_assess.py`).
- [x] All existing test suites pass without regression (204 passed, 5 skipped).

Evidence: 20/20 new tests pass in 0.90s; 204/204 total test suite pass in 55.36s.

Validation commands:
- `.venv\Scripts\python.exe -m pytest ml/tests/test_cost_engine.py ml/tests/test_decision.py ml/tests/test_pipeline_assess.py -v`
- `.venv\Scripts\python.exe -m pytest ml/tests/ -q`

---

### BE-001 — FastAPI Backend Foundation & Assessment APIs

- Phase: 13 (Backend Foundation & Assessment APIs)
- Owner: Member 5 (Backend) / Antigravity
- Reviewer: Member 1
- Status: DONE
- Priority: P0
- Dependencies: INF-001 DONE (unified assess_claim pipeline available)
- Files allowed: backend/, uploads/, docs/agent-work-log.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md
- Files prohibited: ml/ (except importing), frontend/
- Objective: implement the FastAPI backend service with in-memory claim store, upload security validation, all 14 REST endpoints, assessment orchestration, reviewer queue, and summary analytics.

Acceptance criteria:
- [x] `backend/app/store.py` implements thread-safe in-memory store for claims, images, assessments, reviews, and timeline events.
- [x] `backend/app/schemas/` defines Pydantic v2 models for claim creation, update, read, assessment response, review correction, and decision.
- [x] `backend/app/services/upload.py` enforces upload security (MIME whitelist, max 10MB, PIL decode verification, SHA-256 computation).
- [x] All 14 API endpoints implemented under `/api/v1`:
  - `GET /health`
  - `POST /claims`, `PATCH /claims/{id}`, `GET /claims/{id}`, `GET /claims`
  - `POST /claims/{id}/images`, `DELETE /images/{id}`
  - `POST /claims/{id}/submit`
  - `POST /claims/{id}/assess`, `GET /assessments/{id}/status`, `GET /claims/{id}/assessment`, `GET /claims/{id}/timeline`
  - `GET /reviews/queue`, `PATCH /reviews/{id}`, `POST /reviews/{id}/decision`
  - `GET /dashboard/summary`
- [x] Claim state machine enforces valid transitions and records timeline status events.
- [x] Unit & integration tests in `backend/tests/` verify health, upload security, full claim lifecycle, and reviewer workflow.
- [x] All backend tests pass (14 passed in 1.05s).

Evidence: 14/14 tests pass in backend/tests/; full create -> upload -> submit -> assess -> review lifecycle verified.

Validation commands:
- `.venv\Scripts\python.exe -m pytest backend/tests/ -v`

---

### FE-001 — Next.js Customer UI (Policyholder Journey)

- Phase: 15 (Customer UI)
- Owner: Member 6 (Frontend) / Antigravity
- Reviewer: Member 1 / Member 7
- Status: DONE
- Priority: P0
- Dependencies: BE-001 DONE (FastAPI backend endpoints and upload handler available)
- Files allowed: frontend/src/, docs/agent-work-log.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md
- Files prohibited: backend/, ml/
- Objective: implement the customer-facing claim filing and assessment flow using Next.js App Router, TypeScript strict mode, and Tailwind CSS. Connect directly to backend APIs at `http://127.0.0.1:8000/api/v1`.

Acceptance criteria:
- [x] `frontend/src/types/` defines TypeScript interfaces for Claim, ClaimImage, TimelineEvent, and Assessment matching the backend schemas.
- [x] `frontend/src/lib/api.ts` provides a typed fetch client for all backend endpoints with error handling.
- [x] `frontend/src/components/` implements reusable UI components: Navbar, Button, Card, Badge, Spinner, Alert, ImageUploader, StepIndicator, AssessmentCard, and DamageOverlayViewer.
- [x] Multi-step claim creation wizard (`/claims/new`):
  - Step 1: Policy and vehicle details with Zod validation.
  - Step 2: Guided drag-and-drop image upload with live thumbnail previews and remove actions.
  - Step 3: Review and declaration before submission.
- [x] Live assessment processing screen (`/claims/[id]/processing`):
  - Polls `/api/v1/assessments/{id}/status` every 1.5s with animated progress steps.
  - Auto-redirects to assessment result screen upon completion.
- [x] Assessment result screen (`/claims/[id]/result`):
  - Displays route badge (e.g. Fast-Track Eligible, Fraud Review, Manual Review).
  - Displays fraud risk meter, severity tier, damaged-part location badge, and estimated cost range (in INR).
  - Shows side-by-side original evidence photograph and toggleable damage bounding boxes.
- [x] Claim timeline screen (`/claims/[id]/timeline`):
  - Chronological audit trail of claim state transitions.
- [x] Customer claims list (`/claims`):
  - List of active claims with status badges and links to results/timelines.
- [x] Landing page (`/`) updated with navigation, feature overview, and "File a Claim" CTA.
- [x] `npm run build` and `npm run typecheck` succeed with zero TypeScript or lint errors.

Evidence: `npm run typecheck` passed with 0 errors; `next build` generated 6 pages cleanly.

Validation commands:
- `cd frontend; npm run typecheck`
- `cd frontend; npm run build`

---

### FE-002 — Reviewer Dashboard & Analytics (Adjuster Workspace)

- Phase: 16 (Reviewer UI & Analytics)
- Owner: Member 6 (Frontend) / Antigravity
- Reviewer: Member 1 / Member 7
- Status: DONE
- Priority: P0
- Dependencies: FE-001 DONE, BE-001 DONE (Review and Dashboard endpoints available)
- Files allowed: frontend/src/, docs/agent-work-log.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md
- Files prohibited: backend/, ml/
- Objective: implement the adjuster workspace for reviewing flagged claims, inspecting side-by-side evidence with YOLO bounding box overlays, applying manual severity/part overrides without overwriting AI outputs, recording final decisions, and viewing executive KPI charts using Recharts.
- Inputs:
  - `/api/v1/reviews/queue`
  - `/api/v1/reviews/{id}`
  - `/api/v1/reviews/{id}/decision`
  - `/api/v1/dashboard/summary`

Acceptance criteria:
- [x] Reviewer API client methods added in `frontend/src/lib/api.ts` (`getReviewQueue`, `saveReviewCorrection`, `submitReviewDecision`, `getDashboardSummary`).
- [x] Review Queue page (`/reviewer/queue`):
  - Displays claims requiring human review (`FRAUD_REVIEW`, `MANUAL_DAMAGE_REVIEW`, etc.).
  - Filterable by route and status; sortable by date and cost.
  - Quick-action link to inspect claim workspace.
- [x] Claim Review Workspace (`/reviewer/claims/[id]`):
  - Side-by-side visual evidence viewer with toggleable YOLO detection bounding boxes.
  - AI Model Insights panel detailing fraud score, severity classification, and dominant damaged part (Location CNN).
  - Itemized repair/replacement cost table.
  - Reviewer Correction Form allowing manual override of severity tier, damaged parts, notes, and final decision (`APPROVED`, `REJECTED`).
  - Saves corrections via `PATCH /api/v1/reviews/{id}` and finalizes decisions via `POST /api/v1/reviews/{id}/decision` without mutating original AI outputs.
- [x] Executive Dashboard (`/reviewer/dashboard`):
  - KPI summary cards (Total claims, Fast-Track rate, Flagged claims, Average cost range, Adjuster override rate).
  - Recharts visualizations: Route distribution pie chart, severity breakdown bar chart, and damaged parts bar chart.
- [x] Responsive layout verified (no horizontal scrolling at 1366x768).
- [x] `npm run typecheck` and `npm run build` succeed with 0 errors.

Evidence: `npm run typecheck` passed (0 errors); `next build` compiled all 10 customer and reviewer routes cleanly.

Validation commands:
- `cd frontend; npm run typecheck`
- `cd frontend; npm run build`

---

### INT-001 — Full-Stack Model Integration, SQLite Database & Model Benchmark Hub

- Phase: 17 (Integration & Benchmarks)
- Owner: All Members / Antigravity
- Reviewer: Team Lead
- Status: DONE
- Priority: P0
- Dependencies: FE-002 DONE, BE-001 DONE
- Files allowed: models/, backend/, frontend/, config/, scripts/, docs/, TASKS.md, TASK_LOCKS.md, PROJECT_STATUS.md
- Objective: integrate all trained team models from the Google Drive zip into a flat `models/` directory; replace in-memory storage with persistent SQLite via SQLAlchemy 2.0; implement the Model Benchmark & Comparison Hub (`/reviewer/models`); upgrade evidence viewer with an interactive before/after split slider.

Acceptance criteria:
- [x] Models extracted into flat `models/` directory (Fraud, Severity CNN, Severity MobileNetV2, ViT-Tiny, Location MobileNetV2, Location EfficientNet, YOLO Damage, YOLO Parts).
- [x] `scripts/verify_models.py` verifies model loading and forward pass on CPU.
- [x] `config/project.yaml` and `assess.py` configured to load `.pt` weights from `models/`.
- [x] Persistent SQLite database (`backend/claimvision.db`) implemented with SQLAlchemy 2.0 ORM models and `DatabaseClaimStore`.
- [x] All 14 backend test cases pass against SQLite.
- [x] Backend benchmark endpoints (`GET /api/v1/models/benchmark`, `POST /api/v1/models/active`) implemented.
- [x] Frontend Model Benchmark Hub (`/reviewer/models`) created with side-by-side comparison tables, winner badges, and live model switcher.
- [x] `DamageOverlayViewer.tsx` upgraded with an interactive before/after split slider and hoverable detection chips.
- [x] Navigation bar updated with "Model Benchmarks" link.
- [x] Frontend `npm run typecheck` and `npm run build` succeed with 0 errors.

Evidence: All 14 backend tests pass with SQLite; `verify_models.py` passes on CPU; Next.js builds all 11 routes cleanly.

Validation commands:
- `.venv\Scripts\python.exe scripts\verify_models.py`
- `.venv\Scripts\pytest.exe backend/tests/ -v`
- `cd frontend; npm run typecheck`
- `cd frontend; npm run build`






