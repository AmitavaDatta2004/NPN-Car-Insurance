# ClaimVision Task Board

Status values: `BACKLOG`, `READY`, `LOCKED`, `IN_PROGRESS`, `REVIEW`, `BLOCKED`, `DONE`.

## Task template

Copy this section for every task. A task may not enter `READY` without acceptance criteria.

```markdown
### TASK-ID â€” Short title

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

### CFG-001 â€” Install repository configuration pack

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

Commit: `787c999` â€” feat: Add initial project configuration and CI/CD setup

---

### CFG-002 â€” Create application skeleton

- Phase: 0
- Owner: Member 1 (executed by Antigravity)
- Reviewer: Members 5 and 6
- Status: DONE
- Priority: P0
- Dependencies: CFG-001
- Files allowed: `backend/`, `frontend/`, `ml/`, `notebooks/`, `data/`, `artifacts/`, `scripts/`, `annotated/`, `reports/`, `demo/`, `docs/agent-work-log.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `config/project.yaml`, `.github/workflows/ci.yml`, `PROJECT_STATUS.md`, `TASKS.md`
- Files prohibited: `README.md`, `AGENTS.md`, `.env.example`, `.gitignore`, `.gitattributes`, `.editorconfig`, `.pre-commit-config.yaml`, all existing `docs/` templates, both `prompts/`
- Objective: create the repository structure defined in `README.md` Â§6 with working backend and frontend placeholders and an importable ML package
- Inputs: `README.md` Â§6 directory structure, `config/project.example.yaml`, `.github/workflows/ci.yml`

Acceptance criteria:

- [x] Full directory skeleton from README Â§6 exists with `.gitkeep` files where needed.
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

Commit: `abf99fb` â€” Add initial implementation of claimvision_ml package with submodules for fraud detection, severity classification, and damage detection

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

### DATA-001 â€” Validate fraud dataset feasibility

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

### ML-001 â€” Train and evaluate fraud baseline

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
- [x] PR-AUC, class metrics, confusion matrix, calibration, and threshold trade-off shown (Val PR-AUC 0.4999, Test PR-AUC 0.5464, sweep 0.20–0.80).
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

### CV-001 — OpenCV evidence-integrity runtime checker

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
- [x] Corrupt/unreadable → `MORE_EVIDENCE_REQUIRED`.
- [x] Resolution below 224×224 → `MORE_EVIDENCE_REQUIRED`.
- [x] Blurry/dark/overexposed/low-contrast → `rejection_reasons` populated.
- [x] Exact SHA-256 duplicate → `DUPLICATE_REVIEW`.
- [x] Near dHash duplicate (Hamming ≤ 4) → `DUPLICATE_REVIEW`.
- [x] EXIF absent → `warnings["exif_absent"]` only; never sets `passed=False` or `route=FRAUD_REVIEW`.
- [x] `draw_bounding_boxes()` and `save_annotated_image()` work and preserve original.
- [x] 19 unit tests pass; full suite (56 tests) passes with no regression.
- [x] Notebook 04 has 12 sections using real Colab dataset images.
- [x] `ruff check ml/` — only 1 pre-existing Phase 2 warning remains (not our files).

Evidence: 19 new tests pass (56 total); `quality/runtime_checker.py` and extended `image_checks.py` committed; notebook 04 fully implemented with real dataset path configuration.

Validation commands:

- `.venv\Scripts\pytest.exe ml/tests/test_quality_runtime.py -v`
- `.venv\Scripts\pytest.exe ml/tests/ -q`
- Run `notebooks/04_opencv_quality_and_integrity.ipynb` top-to-bottom in Colab

