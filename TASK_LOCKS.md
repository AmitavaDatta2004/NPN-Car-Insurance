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

ML-IMPROVE-001: Team lead / Codex; ACTIVE from 2026-09-22 until pushed verification; base 72d4658 after team synchronization.
Scope: severity/detection modules and tests, their training scripts, notebooks 06–12 (including improvement comparison notebooks), README.md, related model cards, docs/EXPERIMENT_LOG.md, docs/agent-work-log.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md. Fraud excluded. Team leader assigned work in this conversation, explicitly instructed pulling the overlapping commits and then CONTINUE. This authorizes reconciliation of the improvement patch with DET-PART-001 shared scope while preserving its implementation and lock. Colab provides datasets and training runtime.

| Task ID | Owner | Scope/files | Started IST | Expected release IST | Base commit | State |
| --- | --- | --- | --- | --- | --- | --- |
| Example only | — | — | — | — | — | RELEASED |
| CFG-002 | Antigravity | backend/, frontend/, ml/, notebooks/, data/, artifacts/, scripts/, annotated/, reports/, demo/, docs/agent-work-log.md, CHANGELOG.md, CONTRIBUTING.md, config/project.yaml, .github/workflows/ci.yml, PROJECT_STATUS.md, TASKS.md | 2026-09-20 14:16 | 2026-09-20 16:15 | 787c999 | RELEASED |
| DATA-001 | Member 2 / Antigravity | ml/src/claimvision_ml/data/, ml/src/claimvision_ml/quality/, notebooks/01_fraud_dataset_audit.ipynb, data/manifests/, docs/DATASET_CARD_FRAUD.md, ml/tests/test_data_audit.py, docs/agent-work-log.md, PROJECT_STATUS.md, TASKS.md, ml/requirements.txt, .gitignore | 2026-09-20 16:20 | 2026-09-20 18:30 | abf99fb | RELEASED |
| ML-001 | Member 2 / Antigravity | ml/src/claimvision_ml/fraud/, ml/tests/test_fraud_model.py, ml/artifacts/fraud/, ml/results/fraud/, notebooks/02_fraud_mobilenetv2_training.ipynb, notebooks/03_fraud_evaluation_and_threshold.ipynb, docs/MODEL_CARD_FRAUD_MNV2_V1.md, docs/EXPERIMENT_LOG.md, docs/agent-work-log.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md | 2026-09-20 19:27 | 2026-09-20 22:00 | de7100c | RELEASED |
| DATA-001-FIX | Antigravity | notebooks/01_fraud_dataset_audit.ipynb, TASK_LOCKS.md | 2026-09-21 00:30 | 2026-09-21 01:00 | de7f653 | RELEASED |
| ML-PATH-FIX | Antigravity | ml/src/claimvision_ml/fraud/dataset.py, notebooks/02_fraud_mobilenetv2_training.ipynb, notebooks/03_fraud_evaluation_and_threshold.ipynb, data/manifests/, TASK_LOCKS.md | 2026-09-21 00:45 | 2026-09-21 01:30 | de7f653 | RELEASED |
| CV-001 | Member 3 / Antigravity | ml/src/claimvision_ml/quality/, ml/tests/test_quality_runtime.py, notebooks/04_opencv_quality_and_integrity.ipynb, docs/agent-work-log.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md | 2026-09-21 01:37 | 2026-09-21 04:00 | de7f653 | RELEASED |
| ML-DOWNLOAD-HELPER | Antigravity | notebooks/02_fraud_mobilenetv2_training.ipynb, TASK_LOCKS.md | 2026-09-21 16:03 | 2026-09-21 16:30 | de7f653 | RELEASED |
| SDATA-001 | Member 3 / Antigravity | ml/src/claimvision_ml/data/severity_audit.py, ml/src/claimvision_ml/data/__init__.py, ml/tests/test_severity_audit.py, ml/tests/conftest.py, ml/pyproject.toml, notebooks/05_severity_dataset_audit.ipynb, data/manifests/severity_*, docs/DATASET_CARD_SEVERITY.md, ml/results/severity/, docs/agent-work-log.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md | 2026-09-21 16:15 | 2026-09-21 18:30 | b1704c6 | RELEASED |
| ML-003 | Member 2 / Antigravity | ml/src/claimvision_ml/fraud/, ml/tests/test_balanced_sampler.py, notebooks/02b_fraud_balanced_resampling_comparison.ipynb, ml/artifacts/fraud/balanced/, ml/results/fraud/balanced_*, docs/EXPERIMENT_LOG.md, docs/MODEL_CARD_FRAUD_MNV2_V1.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md, docs/agent-work-log.md | 2026-09-21 17:25 | 2026-09-21 21:00 | 28debd5 | RELEASED |
| SEV-MNV2-001 | Friend 2 / Antigravity | ml/src/claimvision_ml/severity/mobilenet.py, ml/src/claimvision_ml/severity/dataset.py, ml/src/claimvision_ml/severity/predict.py, ml/src/claimvision_ml/severity/__init__.py, ml/tests/test_severity_mobilenet.py, notebooks/07_severity_mobilenetv2_training.ipynb, docs/MODEL_CARD_SEVERITY_MNV2_V1.md, docs/EXPERIMENT_LOG.md, docs/agent-work-log.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md | 2026-09-21 17:47 | 2026-09-21 20:30 | 28debd5 | RELEASED |
| SEV-VIT-001 | Member 4 / Antigravity | ml/src/claimvision_ml/severity/vit.py, ml/src/claimvision_ml/severity/__init__.py, ml/tests/test_severity_vit.py, notebooks/08_severity_vit_tiny_training.ipynb, scripts/build_notebook_08.py, scripts/run_severity_vit.py, docs/MODEL_CARD_SEVERITY_VIT_TINY.md, ml/results/severity/vit/, ml/artifacts/severity/vit/, docs/EXPERIMENT_LOG.md, docs/agent-work-log.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md | 2026-09-21 17:50 | 2026-09-21 21:00 | 28debd5 | RELEASED |
| SEV-CNN-001 | Member 3 / Antigravity | notebooks/06_severity_cnn_training.ipynb, ml/src/claimvision_ml/severity/cnn.py, ml/src/claimvision_ml/severity/__init__.py, ml/tests/test_severity_cnn.py, ml/results/severity/cnn_metrics.json, docs/EXPERIMENT_LOG.md, docs/agent-work-log.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md | 2026-09-21 17:51 | 2026-09-21 21:00 | 28debd5 | RELEASED |
| DET-COCO-001 | Member 4 / Antigravity | ml/src/claimvision_ml/detection/, ml/tests/test_coco_converter.py, ml/results/detection/, notebooks/10_coco_annotation_audit_and_conversion.ipynb, docs/EXPERIMENT_LOG.md, docs/agent-work-log.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md | 2026-09-21 23:55 | 2026-09-22 03:00 | 1231ecf | RELEASED |
| DET-YOLO-001 | Member 4 / Antigravity | ml/src/claimvision_ml/detection/, ml/tests/test_damage_detector.py, notebooks/11_yolo_damage_training.ipynb, docs/MODEL_CARD_DAMAGE_YOLO_V1.md, docs/EXPERIMENT_LOG.md, docs/agent-work-log.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md | 2026-09-22 01:46 | 2026-09-22 04:30 | 6884cc7 | RELEASED |
| DET-PART-001 | Member 4 / Antigravity | ml/src/claimvision_ml/detection/, ml/tests/test_part_detector.py, notebooks/12_yolo_part_training.ipynb, docs/MODEL_CARD_PART_YOLO_V1.md, docs/EXPERIMENT_LOG.md, docs/agent-work-log.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md | 2026-09-22 02:19 | 2026-09-22 05:00 | a52ed77 | ACTIVE |


## Lock template


```markdown
| TASK-ID | Member name | Exact files/directories | YYYY-MM-DD HH:MM | YYYY-MM-DD HH:MM | short hash | ACTIVE |
```

## Collision procedure

If two tasks require the same file:

1. Stop immediately.
2. Inspect the current lock in this file.
3. Message the lock owner with task ID and reason.
4. If authorized, the owner appends an explicit delegation note below the table.
5. If denied, select an unblocked task or wait.
