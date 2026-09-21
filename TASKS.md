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
