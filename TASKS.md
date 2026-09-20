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
- Status: READY
- Priority: P0
- Dependencies: Root `README.md` approved
- Files allowed: configuration files supplied in this pack
- Objective: establish one operating contract for all laptops and agents

Acceptance criteria:

- [ ] Pack files exist at repository root using the documented paths.
- [ ] Seven member names replace placeholders.
- [ ] `.env` is ignored and `.env.example` is committed.
- [ ] All members confirm they read `AGENTS.md`.
- [ ] Configuration commit is pushed to `main`.

### CFG-002 — Create application skeleton

- Phase: 0
- Owner: Member 1
- Reviewer: Members 5 and 6
- Status: BACKLOG
- Priority: P0
- Dependencies: CFG-001
- Files allowed: `apps/`, `ml/`, `tests/`, package manifests
- Objective: create the repository structure defined in `README.md`

Acceptance criteria:

- [ ] Next.js application runs locally.
- [ ] FastAPI health endpoint returns success.
- [ ] Python ML package imports successfully.
- [ ] No dataset or weights are committed.
- [ ] Setup is independently verified on two team laptops.

### DATA-001 — Validate fraud dataset feasibility

- Phase: 1
- Owner: Member 2
- Reviewer: Member 1
- Status: BACKLOG
- Priority: P0
- Dependencies: CFG-002
- Files allowed: fraud audit notebook, manifests, dataset card, reusable audit utilities
- Objective: determine whether the proposed labels support an honest suspicious-image classifier

Acceptance criteria:

- [ ] Dataset source, licence, schema, counts, duplicates, corrupt images, and class balance documented.
- [ ] Group-aware frozen manifests created.
- [ ] Source and watermark shortcut risks examined.
- [ ] Representative images and failure cases displayed.
- [ ] Written go/modify/stop recommendation recorded.

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

