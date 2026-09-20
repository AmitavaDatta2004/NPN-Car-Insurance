# Changelog

All notable changes to ClaimVision AI are recorded here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: prototype phases, not semantic versions.

---

## [Phase 0] — 2026-09-20

### Added

- **Repository skeleton** — complete directory structure per README §6:
  `backend/`, `frontend/`, `ml/`, `notebooks/`, `data/`, `artifacts/`, `scripts/`, `annotated/`, `reports/`, `demo/`
- **`claimvision_ml` Python package** — installable ML package at `ml/src/claimvision_ml/` with sub-packages:
  `data`, `quality`, `fraud`, `severity`, `detection`, `costing`, `pipeline`
- **FastAPI backend placeholder** — `backend/main.py` with `/api/v1/health` endpoint; all app sub-packages stubbed
- **Next.js frontend placeholder** — App Router + TypeScript + Tailwind CSS landing page at `frontend/`
- **16 notebook stubs** — `notebooks/00_` through `notebooks/15_` with title, purpose, and owner metadata cells
- **`CHANGELOG.md`** — this file
- **`CONTRIBUTING.md`** — contribution guide
- **`docs/agent-work-log.md`** — agent-assisted work log (required by README §7.1)
- **`config/project.yaml`** — committed portable project configuration (copy of `project.example.yaml`)
- **`.gitkeep` files** — in `data/manifests/`, `data/samples/`, `artifacts/models/`, `artifacts/runs/`, `artifacts/reports/`, `ml/artifacts/`, `ml/results/`, `demo/fixtures/`, `demo/expected-results/`

### Fixed

- **CI workflow paths** — corrected `apps/api/` → `backend/` and `apps/web/` → `frontend/` throughout `.github/workflows/ci.yml`

### Changed

- **`PROJECT_STATUS.md`** — Phase 0 marked In progress, sprint objectives populated
- **`TASKS.md`** — CFG-001 marked DONE; CFG-002 expanded with full acceptance criteria and marked IN_PROGRESS
- **`TASK_LOCKS.md`** — CFG-002 active lock added

---

## [Configuration pack] — 2026-09-20 (commit 787c999)

### Added

- `AGENTS.md`, `PROJECT_STATUS.md`, `TASKS.md`, `TASK_LOCKS.md`
- `docs/DECISIONS.md`, `docs/EXPERIMENT_LOG.md`, `docs/DATASET_CARD_TEMPLATE.md`, `docs/MODEL_CARD_TEMPLATE.md`, `docs/NOTEBOOK_REVIEW_CHECKLIST.md`, `docs/DEMO_RUNBOOK.md`
- `prompts/ANTIGRAVITY_TASK_PROMPT.md`, `prompts/CODEX_TASK_PROMPT.md`
- `.env.example`, `.gitignore`, `.gitattributes`, `.editorconfig`, `.pre-commit-config.yaml`
- `.github/ISSUE_TEMPLATE/task.yml`, `.github/workflows/ci.yml`
- `config/project.example.yaml`
- `README_INSTALL.md`

---

## [Initial] — 2026-09-20 (commit e2ccabc)

### Added

- `README.md` — complete implementation specification (1 568 lines)
