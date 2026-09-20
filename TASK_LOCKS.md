# Active Task Locks

This file substitutes for branch isolation because all seven members work directly on `main`.

## Rules

1. Add a lock before editing.
2. One active editor per file or directory boundary.
3. A lock expires only at the listed time or when explicitly released.
4. Do not delete another member's lock.
5. If a lock is stale, contact the owner and team leader before takeover.
6. Release only after commit, push, remote verification, and status update.

## Active locks

| Task ID | Owner | Scope/files | Started IST | Expected release IST | Base commit | State |
| --- | --- | --- | --- | --- | --- | --- |
| Example only | — | — | — | — | — | RELEASED |
| CFG-002 | Antigravity | backend/, frontend/, ml/, notebooks/, data/, artifacts/, scripts/, annotated/, reports/, demo/, docs/agent-work-log.md, CHANGELOG.md, CONTRIBUTING.md, config/project.yaml, .github/workflows/ci.yml, PROJECT_STATUS.md, TASKS.md | 2026-09-20 14:16 | 2026-09-20 16:15 | 787c999 | RELEASED |
| DATA-001 | Member 2 / Antigravity | ml/src/claimvision_ml/data/, ml/src/claimvision_ml/quality/, notebooks/01_fraud_dataset_audit.ipynb, data/manifests/, docs/DATASET_CARD_FRAUD.md, ml/tests/test_data_audit.py, docs/agent-work-log.md, PROJECT_STATUS.md, TASKS.md, ml/requirements.txt, .gitignore | 2026-09-20 16:20 | 2026-09-20 18:30 | abf99fb | RELEASED |
| ML-001 | Member 2 / Antigravity | ml/src/claimvision_ml/fraud/, ml/tests/test_fraud_model.py, ml/artifacts/fraud/, ml/results/fraud/, notebooks/02_fraud_mobilenetv2_training.ipynb, notebooks/03_fraud_evaluation_and_threshold.ipynb, docs/MODEL_CARD_FRAUD_MNV2_V1.md, docs/EXPERIMENT_LOG.md, docs/agent-work-log.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md | 2026-09-20 19:27 | 2026-09-20 22:00 | de7100c | RELEASED |
| DATA-001-FIX | Antigravity | notebooks/01_fraud_dataset_audit.ipynb, TASK_LOCKS.md | 2026-09-21 00:30 | 2026-09-21 01:00 | de7f653 | RELEASED |
| ML-PATH-FIX | Antigravity | ml/src/claimvision_ml/fraud/dataset.py, notebooks/02_fraud_mobilenetv2_training.ipynb, notebooks/03_fraud_evaluation_and_threshold.ipynb, data/manifests/, TASK_LOCKS.md | 2026-09-21 00:45 | 2026-09-21 01:30 | de7f653 | RELEASED |

## Lock template

```markdown
| TASK-ID | Member name | Exact files/directories | YYYY-MM-DD HH:MM | YYYY-MM-DD HH:MM | short hash | ACTIVE |
```

## Collision procedure

If two tasks require the same file:

1. Pause the later task.
2. Decide which change lands first.
3. First owner commits and pushes.
4. Second owner pulls with rebase, reruns relevant tests, then takes the lock.
5. Record any contract change in `docs/DECISIONS.md`.

