# ClaimVision Task Board

Status values: `BACKLOG`, `READY`, `LOCKED`, `IN_PROGRESS`, `REVIEW`, `BLOCKED`, `DONE`.

## Task template

Copy this section for every task. A task may not enter `READY` without acceptance criteria.

```markdown
### TASK-ID — Short title

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

### CFG-001 — Install repository configuration pack

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

Commit: `787c999` — feat: Add initial project configuration and CI/CD setup

---

### CFG-002 — Create application skeleton

- Phase: 0
- Owner: Member 1 (executed by Antigravity)
- Reviewer: Members 5 and 6
- Status: DONE
- Priority: P0
- Dependencies: CFG-001
- Files allowed: `backend/`, `frontend/`, `ml/`, `notebooks/`, `data/`, `artifacts/`, `scripts/`, `annotated/`, `reports/`, `demo/`, `docs/agent-work-log.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `config/project.yaml`, `.github/workflows/ci.yml`, `PROJECT_STATUS.md`, `TASKS.md`
- Files prohibited: `README.md`, `AGENTS.md`, `.env.example`, `.gitignore`, `.gitattributes`, `.editorconfig`, `.pre-commit-config.yaml`, all existing `docs/` templates, both `prompts/`
- Objective: create the repository structure defined in `README.md` §6 with working backend and frontend placeholders and an importable ML package
- Inputs: `README.md` §6 directory structure, `config/project.example.yaml`, `.github/workflows/ci.yml`

Acceptance criteria:

- [x] Full directory skeleton from README §6 exists with `.gitkeep` files where needed.
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

Commit: `abf99fb` — Add initial implementation of claimvision_ml package with submodules for fraud detection, severity classification, and damage detection

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

### DATA-001 — Validate fraud dataset feasibility

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

### ML-001 — Train and evaluate fraud baseline

- Phase: 2
- Owner: Member 2
- Reviewer: Members 1 and 3
- Status: BACKLOG
- Priority: P0
- Dependencies: DATA-001 accepted
- Objective: produce a calibrated suspicious-image signal before severity work begins

Acceptance criteria:

- [ ] MobileNetV2 experiment is reproducible.
- [ ] PR-AUC, class metrics, confusion matrix, calibration, and threshold trade-off shown.
- [ ] Untouched test set used exactly once for final reporting.
- [ ] Exported artifact passes smoke inference.
- [ ] Model card clearly limits the meaning of fraud output.

## Assignment rule

Only move one task per member into `IN_PROGRESS` at a time. Add the lock first, then update its status. When complete, link the commit and evidence under the task before marking `DONE`.
