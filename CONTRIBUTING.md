# Contributing to ClaimVision AI

## Before you do anything

1. Read [`README.md`](README.md) completely. It is the controlling implementation specification.
2. Read [`AGENTS.md`](AGENTS.md). It is the binding operating contract for all contributors and coding agents.
3. Read [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for the current phase and blockers.
4. Read [`TASKS.md`](TASKS.md) and [`TASK_LOCKS.md`](TASK_LOCKS.md) before touching any file.

## Workflow

This project uses **direct commits to `main`**. There are no feature branches.
Collision risk is managed through task locks, single-file ownership, pull-rebase, and small atomic commits.

### Starting a task

1. Verify the task has an ID and acceptance criteria in `TASKS.md`.
2. Confirm prerequisites are complete.
3. Add an active lock row in `TASK_LOCKS.md`.
4. Announce the task ID to the team.
5. Run `git pull --rebase origin main`.
6. Run `git status --short` — must be clean.

### During work

- Change only the files listed in your task lock.
- Keep commits small and atomic.
- Do not combine unrelated formatting or refactoring changes.
- Update documentation in the same task when behaviour or contracts change.
- Put reusable ML logic in `ml/src/claimvision_ml/`, not inside notebooks.

### Before committing

1. Re-read acceptance criteria.
2. Run the smallest relevant tests plus lint checks.
3. Run `git diff --check` and inspect `git diff`.
4. Confirm no secret, dataset, weight, cache, or notebook checkpoint is staged.
5. Update `PROJECT_STATUS.md`, `TASKS.md`, `CHANGELOG.md`, and `docs/agent-work-log.md`.

### Commit format

```
<type>(<module>): <task-id> <short result>
```

Examples:

```
data(fraud): P1-T2 add CSV and missing-file audit
model(severity): P7-T3 evaluate ViT-Tiny checkpoint
feat(backend): P13-T2 persist assessment state transitions
test(frontend): P16-T4 add claim submission E2E scenario
```

## Prohibited actions

- `git push --force` or `git push --force-with-lease`
- Amending another member's published commit
- Committing secrets, local paths, datasets, model weights, or caches
- Bypassing or removing a failing test to get green CI
- Fabricating metrics, dataset statistics, or model results

## Coding agents (Antigravity, Codex)

Agents follow `AGENTS.md` strictly. The human task owner reviews every diff before committing.
Use `prompts/ANTIGRAVITY_TASK_PROMPT.md` and `prompts/CODEX_TASK_PROMPT.md` to assign tasks.

## Environment setup

```powershell
# Python — use the existing venv
.venv\Scripts\pip.exe install -e ml/
.venv\Scripts\pip.exe install -r backend/requirements-dev.txt

# Frontend
cd frontend
npm install

# Pre-commit hooks (optional but recommended)
.venv\Scripts\pre-commit.exe install
```

See `README.md` §15 for all documented commands.
