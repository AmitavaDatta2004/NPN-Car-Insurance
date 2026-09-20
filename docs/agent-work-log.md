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
