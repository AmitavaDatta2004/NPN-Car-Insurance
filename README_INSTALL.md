# ClaimVision Configuration Pack

Copy the contents of this directory into the root of the ClaimVision AI repository. Do not copy the outer `claimvision-configuration-pack` directory itself.

## Included files

| File | Purpose |
| --- | --- |
| `AGENTS.md` | Binding instructions for Antigravity, Codex, and team members |
| `PROJECT_STATUS.md` | Current phase, health, blockers, and demonstration readiness |
| `TASKS.md` | Task backlog with acceptance criteria and dependencies |
| `TASK_LOCKS.md` | Prevents simultaneous edits while everyone pushes to `main` |
| `docs/DECISIONS.md` | Architecture decision record |
| `docs/EXPERIMENT_LOG.md` | Central ML experiment registry |
| `docs/DATASET_CARD_TEMPLATE.md` | Dataset documentation template |
| `docs/MODEL_CARD_TEMPLATE.md` | Exported model documentation template |
| `docs/NOTEBOOK_REVIEW_CHECKLIST.md` | Judge-facing notebook quality gate |
| `docs/DEMO_RUNBOOK.md` | Repeatable presentation and fallback plan |
| `prompts/ANTIGRAVITY_TASK_PROMPT.md` | Prompt used to assign one task to Antigravity |
| `prompts/CODEX_TASK_PROMPT.md` | Prompt used to assign one task to Codex |
| `.env.example` | Environment-variable contract |
| `.gitignore` | Prevents local data, weights, secrets, caches, and notebook checkpoints from entering Git |
| `.editorconfig` | Common editor formatting rules |
| `.gitattributes` | Line-ending and generated-file handling |
| `.pre-commit-config.yaml` | Local safety and formatting checks |
| `.github/ISSUE_TEMPLATE/task.yml` | Structured GitHub task form |
| `.github/workflows/ci.yml` | Backend, frontend, and repository checks |
| `config/project.example.yaml` | Shared application and ML path configuration |

## Installation

1. Place the root implementation plan at `README.md`.
2. Copy every file from this pack into the repository, preserving paths.
3. Replace placeholder team names in `TASKS.md` and `PROJECT_STATUS.md`.
4. Copy `config/project.example.yaml` to `config/project.yaml`; edit only machine-independent project settings.
5. Copy `.env.example` to `.env`; each member fills only local values.
6. Install pre-commit and run `pre-commit install`.
7. Commit the configuration pack as the first repository configuration commit.
8. Every agent must read `README.md` and `AGENTS.md` before its first task.

## Important

- `config/project.yaml` may be committed when it contains only portable paths and public configuration.
- `.env` must never be committed.
- Dataset files, images, trained weights, and generated runs remain outside Git.
- This pack intentionally uses standard files rather than assuming undocumented Antigravity-only syntax.

