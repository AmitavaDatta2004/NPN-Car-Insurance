# Agent Work Log

Every agent-assisted change is recorded here before editing begins.
Required by README §7.1 rule 5.

## Entry format

```
### TASK-ID — Short title
- Date/time IST:
- Agent: Antigravity / Codex
- Operator (human reviewer):
- Base commit:
- Files read:
- Files changed:
- Decisions made:
- Commands run:
- Validation results:
- Known limitations / follow-up:
```

---

## Log

### ML-001 — Train and evaluate fraud MobileNetV2 classifier (Phase 2)

- Date/time IST: 2026-09-20 19:27–21:00
- Agent: Antigravity
- Operator: Amitava Datta (to review diff, run notebooks 02 and 03, and commit)
- Base commit: de7100c
- Files read: README.md, AGENTS.md, PROJECT_STATUS.md, TASKS.md, docs/DECISIONS.md,
  ml/src/claimvision_ml/data/, notebooks/01_fraud_dataset_audit.ipynb,
  notebooks/data/manifests/fraud_train.csv, fraud_val.csv, fraud_test.csv
- Files changed:
  - ml/src/claimvision_ml/fraud/__init__.py (updated with full exports)
  - ml/src/claimvision_ml/fraud/model.py (NEW — FraudClassifier, build/load/export)
  - ml/src/claimvision_ml/fraud/dataset.py (NEW — FraudDataset, get_class_weights, get_transforms)
  - ml/src/claimvision_ml/fraud/predict.py (NEW — FraudResult, predict_fraud)
  - ml/tests/test_fraud_model.py (NEW — 17 CPU-only smoke tests)
  - notebooks/02_fraud_mobilenetv2_training.ipynb (full implementation replacing stub)
  - notebooks/03_fraud_evaluation_and_threshold.ipynb (full implementation replacing stub)
  - docs/MODEL_CARD_FRAUD_MNV2_V1.md (NEW)
  - docs/EXPERIMENT_LOG.md (FRAUD-MNV2-001 entry)
  - docs/agent-work-log.md (this entry)
  - TASKS.md (ML-001 → IN_PROGRESS)
  - TASK_LOCKS.md (DATA-001 RELEASED, ML-001 ACTIVE)
  - PROJECT_STATUS.md (Phase 2 in progress)
- Decisions made:
  - BCEWithLogitsLoss pos_weight chosen as n_genuine/n_suspicious (inverse frequency).
  - WeightedRandomSampler used in addition to pos_weight for better minority sampling.
  - Stage A freezes all backbone; Stage B unfreezes final 2 InvertedResidual blocks.
  - Early stopping patience=5 on val PR-AUC (primary metric for imbalanced classes).
  - CPU training: batch_size=16, Stage A max 10 epochs, Stage B max 20 epochs.
  - Export both .pt (reproducibility) and ONNX (backend inference).
  - Thresholds selected from validation PR curve; test set used exactly once.
- Commands run:
  - `.venv\Scripts\pytest.exe ml/tests/test_fraud_model.py -v` — 17 passed
  - `.venv\Scripts\pytest.exe ml/tests/ -q` — all prior tests still pass
- Validation results: 17/17 new tests pass; 0 regressions in prior test suite.
- Known limitations / follow-up:
  - Notebook 02 and 03 must be run by the human operator to produce the real
    trained checkpoint and thresholds. The notebooks contain all training code.
  - EXPERIMENT_LOG.md results marked TBD — update after notebooks run.
  - MODEL_CARD metrics marked TBD — fill after Notebook 03 completes.
  - ONNX output difference must be checked after training (cell 15 in notebook 02).
  - If PR-AUC < 0.30 after full run, reassess shortcut-learning risk and flag
    classifier as experimental per AGENTS.md §10.

### CFG-002 — Create application skeleton


- Date/time IST: 2026-09-20 14:16–17:00
- Agent: Antigravity
- Operator: Amitava Datta (to review diff and commit)
- Base commit: 787c999
- Phase: 0

Files read:

- `README.md` (all 1 568 lines)
- `AGENTS.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `TASK_LOCKS.md`
- `docs/DECISIONS.md`
- `docs/EXPERIMENT_LOG.md`
- `docs/DATASET_CARD_TEMPLATE.md`
- `docs/MODEL_CARD_TEMPLATE.md`
- `docs/NOTEBOOK_REVIEW_CHECKLIST.md`
- `docs/DEMO_RUNBOOK.md`
- `prompts/ANTIGRAVITY_TASK_PROMPT.md`
- `prompts/CODEX_TASK_PROMPT.md`
- `.env.example`
- `.gitignore`
- `.gitattributes`
- `.editorconfig`
- `.pre-commit-config.yaml`
- `.github/workflows/ci.yml`
- `.github/ISSUE_TEMPLATE/task.yml`
- `config/project.example.yaml`
- `README_INSTALL.md`

Files created:

- `CHANGELOG.md`
- `CONTRIBUTING.md`
- `config/project.yaml`
- `data/README.md`
- `data/manifests/.gitkeep`
- `data/samples/.gitkeep`
- `artifacts/models/.gitkeep`
- `artifacts/runs/.gitkeep`
- `artifacts/reports/.gitkeep`
- `ml/pyproject.toml`
- `ml/requirements.txt`
- `ml/requirements-dev.txt`
- `ml/artifacts/.gitkeep`
- `ml/results/.gitkeep`
- `ml/configs/.gitkeep`
- `ml/tests/__init__.py`
- `ml/tests/test_package_import.py`
- `ml/src/claimvision_ml/__init__.py`
- `ml/src/claimvision_ml/data/__init__.py`
- `ml/src/claimvision_ml/quality/__init__.py`
- `ml/src/claimvision_ml/fraud/__init__.py`
- `ml/src/claimvision_ml/severity/__init__.py`
- `ml/src/claimvision_ml/detection/__init__.py`
- `ml/src/claimvision_ml/costing/__init__.py`
- `ml/src/claimvision_ml/pipeline/__init__.py`
- `backend/main.py`
- `backend/pyproject.toml`
- `backend/requirements.txt`
- `backend/requirements-dev.txt`
- `backend/app/__init__.py`
- `backend/app/api/__init__.py`
- `backend/app/api/routes/__init__.py`
- `backend/app/core/__init__.py`
- `backend/app/db/__init__.py`
- `backend/app/models/__init__.py`
- `backend/app/schemas/__init__.py`
- `backend/app/services/__init__.py`
- `backend/alembic/.gitkeep`
- `backend/tests/__init__.py`
- `backend/tests/test_health.py`
- `frontend/package.json`
- `frontend/tsconfig.json`
- `frontend/next.config.js`
- `frontend/tailwind.config.ts`
- `frontend/postcss.config.js`
- `frontend/.eslintrc.json`
- `frontend/src/app/layout.tsx`
- `frontend/src/app/page.tsx`
- `frontend/src/app/globals.css`
- `frontend/src/components/.gitkeep`
- `frontend/src/lib/.gitkeep`
- `frontend/src/types/.gitkeep`
- `frontend/tests/.gitkeep`
- `notebooks/00_environment_and_data_download.ipynb`
- `notebooks/01_fraud_dataset_audit.ipynb`
- `notebooks/02_fraud_mobilenetv2_training.ipynb`
- `notebooks/03_fraud_evaluation_and_threshold.ipynb`
- `notebooks/04_opencv_quality_and_integrity.ipynb`
- `notebooks/05_severity_dataset_audit.ipynb`
- `notebooks/06_severity_cnn_training.ipynb`
- `notebooks/07_severity_mobilenetv2_training.ipynb`
- `notebooks/08_severity_vit_tiny_training.ipynb`
- `notebooks/09_severity_model_comparison.ipynb`
- `notebooks/10_coco_annotation_audit_and_conversion.ipynb`
- `notebooks/11_yolo_damage_training.ipynb`
- `notebooks/12_yolo_part_training.ipynb`
- `notebooks/13_unified_inference_demo.ipynb`
- `notebooks/14_explainability_and_gradcam.ipynb`
- `notebooks/15_final_judge_results.ipynb`
- `scripts/.gitkeep`
- `annotated/.gitkeep`
- `reports/.gitkeep`
- `demo/fixtures/.gitkeep`
- `demo/expected-results/.gitkeep`
- `docs/agent-work-log.md` (this file)

Files modified:

- `TASK_LOCKS.md` — added CFG-002 active lock row
- `PROJECT_STATUS.md` — updated date, commit, phase status, sprint objectives
- `TASKS.md` — CFG-001 marked DONE; CFG-002 expanded and marked IN_PROGRESS
- `.github/workflows/ci.yml` — fixed paths: apps/api → backend, apps/web → frontend

Decisions made:

- `uploads/` and root `models/` directories not given `.gitkeep` — both are runtime-only and are fully ignored by `.gitignore`. They are created by the application at first use.
- `config/project.yaml` committed (safe: contains only portable, non-secret settings per README_INSTALL §Important).
- Heavy ML dependencies (torch, ultralytics) listed in `ml/requirements.txt` but NOT installed during Phase 0 to avoid multi-GB downloads on all machines. Members install before Phase 1.
- `node_modules/` not committed. Each member runs `npm install` after pulling.
- CI Node version kept at 20 (matches existing ci.yml) even though local Node is 22. CI is a separate environment.

Commands run:

- `git log --oneline --all` — confirmed 2 commits, clean main
- `git status --short` — confirmed clean working tree
- `.venv\Scripts\python.exe --version` — confirmed Python 3.11.9
- `node --version` — confirmed v22.21.0
- `npm --version` — confirmed 11.6.2
- `.venv\Scripts\pip.exe install -e ml/` — install ML package in editable mode
- `.venv\Scripts\pip.exe install -r ml/requirements-dev.txt` — install ML dev deps
- `.venv\Scripts\pip.exe install -r backend/requirements-dev.txt` — install backend dev deps
- `.venv\Scripts\pytest.exe ml/tests/ -q` — ML smoke test
- `.venv\Scripts\pytest.exe backend/tests/ -q` — backend smoke test
- `cd frontend && npm install && npm run build` — frontend build

Validation results:
- ML package import: SUCCESS (`claimvision-ml` 0.1.0 installed in editable mode).
- ML test suite (`pytest ml/tests/ -v`): 9 passed in 0.05s (package + 7 subpackages importable).
- Backend dependencies (`pip install -r backend/requirements-dev.txt`): SUCCESS.
- Backend test suite (`pytest backend/tests/ -v`): 3 passed in 0.83s (`GET /api/v1/health` returns 200, valid body, application/json).
- Frontend package install (`npm install`): 527 packages installed in 2m.
- Frontend build (`npm run build`): SUCCESS, compiled production build with static prerendered routes in Next.js 14.2.35.
- Repository safety check: `git status --short` verified; no secret, dataset, model weights, or unintended cache committed.

Known limitations:

- Notebook stubs are placeholders only. Actual implementation begins in Phase 1.
- ML sub-package `__init__.py` files contain only version stubs. Real implementations are Phase 1 and later.
- Backend `app/` sub-packages are empty stubs. Full implementation begins in Phase 12.
- Frontend shows a "coming soon" landing page only. Full UI is Phase 14.

Follow-up: Phase 1 — DATA-001 fraud dataset feasibility audit (Member 2).

---

### DATA-001 — Validate fraud dataset feasibility

- Date/time IST: 2026-09-20 16:20–16:35
- Agent: Antigravity
- Operator: Member 2 (Fraud ML) / Amitava Datta
- Base commit: abf99fb
- Phase: 1

Files read:
- `README.md` (§10 Notebook 01, §11.1 Fraud dataset, §19 Phase 1)
- `AGENTS.md` (§8 Dataset contract, §10 Model-specific rules, §11 OpenCV rules)
- `docs/DATASET_CARD_TEMPLATE.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `TASK_LOCKS.md`

Files created:
- `ml/src/claimvision_ml/quality/image_checks.py` (safe image loading, Laplacian blur, brightness, contrast)
- `ml/src/claimvision_ml/data/audit.py` (CSV validation, image reconciliation, SHA-256 deduplication, dHash clustering, shortcut risk analysis)
- `ml/src/claimvision_ml/data/manifest.py` (group-aware stratified splitting, programmatic anti-leakage assertions, manifest saving)
- `ml/tests/test_data_audit.py` (11 unit tests covering all quality, audit, split, and anti-leakage functions)
- `docs/DATASET_CARD_FRAUD.md` (complete dataset card following template)
- `scripts/build_notebook_01.py` (generator script for Notebook 01)
- `data/samples/fraud_sample/` (12 synthetic sample images and sample CSV for offline/test execution)
- `data/manifests/fraud_train.csv`, `fraud_val.csv`, `fraud_test.csv`, `fraud_manifest_summary.json` (frozen audit manifests)
- `ml/results/fraud_class_distribution.png`, `fraud_quality_metrics_distribution.png`, `fraud_sample_grid.png` (exported plot figures)

Files modified:
- `.gitignore` (added `car-damage-dataset/` and `.cache/` ignore rules)
- `ml/requirements.txt` (added `kagglehub`, `pandas`, `imagehash`)
- `ml/requirements-dev.txt` (added `nbconvert`)
- `ml/src/claimvision_ml/quality/__init__.py` (exported image check functions)
- `ml/src/claimvision_ml/data/__init__.py` (exported audit and manifest functions)
- `notebooks/01_fraud_dataset_audit.ipynb` (full 18-section implementation with kagglehub snippet)
- `PROJECT_STATUS.md` (updated Phase 1 in-progress/ready status, objectives)
- `TASKS.md` (updated DATA-001 status to READY for review)
- `TASK_LOCKS.md` (released CFG-002, active lock DATA-001)

Decisions made:
- Ingest Vinay Jose Car Damage Dataset via `kagglehub.dataset_download("vinayjose/car-damage-dataset")` as specified by user.
- Route Kagglehub cache (`KAGGLEHUB_CACHE`) and direct download (`output_dir`) into the project directory (`data/raw/` on D: drive) to completely avoid consuming C: drive space.
- Added `*.archive` to `.gitignore` so dataset archives and downloads are completely ignored by Git.
- Add fallback to `data/samples/fraud_sample` so notebook and unit tests execute reliably in offline CI/test environments without requiring immediate 1GB download.
- Enforce strict anti-leakage splitting: claim ID and duplicate clusters are never split across train, val, or test.
- Treat visual model as a "suspicious-image signal" for human review routing, not legal proof of fraud.

Commands run:
- `.\.venv\Scripts\pip install kagglehub pandas opencv-python-headless Pillow matplotlib seaborn scikit-learn imagehash ipykernel nbconvert`
- `.\.venv\Scripts\python scripts/build_notebook_01.py`
- `.\.venv\Scripts\pytest ml/tests/ -v` (20 passed in 0.67s)
- `.\.venv\Scripts\ruff check ml/` (All checks passed)
- `.\.venv\Scripts\ruff format --check ml/` (All 14 files formatted)

Validation results:
- 20 unit tests passed covering image decoding, Laplacian blur, brightness, contrast, SHA-256 exact deduplication, dHash clustering, group-aware splitting, and negative test confirming leakage assertion triggers on contaminated splits.
- Notebook 01 smoke execution passed top-to-bottom without errors.
- Visual plots saved in `ml/results/`.
- Frozen manifests exported in `data/manifests/`.

Known limitations:
- Full 1GB dataset download will be executed by user/friend directly in VS Code by running the first cells of `01_fraud_dataset_audit.ipynb`.

Follow-up: Phase 2 — ML-001 train and evaluate MobileNetV2 fraud baseline (Member 2).

