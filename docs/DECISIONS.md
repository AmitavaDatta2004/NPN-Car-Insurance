# Architecture Decision Record

Never rewrite an accepted decision to hide history. Mark it superseded and add a new decision.

## Decision template

### ADR-XXX — Title

- Date:
- Status: Proposed / Accepted / Superseded / Rejected
- Decision owners:
- Related task:

Context:

Describe the problem, constraints, and evidence.

Options considered:

1. Option and trade-offs.
2. Option and trade-offs.

Decision:

State the selected option precisely.

Consequences:

- Positive consequence.
- Cost or limitation.

Validation/revisit trigger:

State what evidence could cause reconsideration.

## Accepted foundation decisions

### ADR-001 — Notebook-first ML evidence

- Date: YYYY-MM-DD
- Status: Accepted
- Decision owners: DEV NEXUS
- Related task: CFG-001

Context:

The project is presented to judges and must make preprocessing, training, evaluation, and failures visible.

Decision:

ML experiments will be demonstrated in committed Jupyter notebooks with curated outputs. Reusable logic will live in importable Python modules so notebooks and FastAPI use the same implementation.

Consequences:

- Judges can inspect the work visually.
- Notebook outputs require deliberate size and privacy control.
- Logic cannot exist only in notebooks.

### ADR-002 — Fraud before severity

- Date: YYYY-MM-DD
- Status: Accepted
- Decision owners: DEV NEXUS
- Related task: DATA-001

Decision:

Fraud dataset auditing and the suspicious-image model are implemented before severity. High-risk or unreliable evidence is routed to human review before automated severity and cost recommendations are finalized.

### ADR-003 — Direct commits to main with task locks

- Date: YYYY-MM-DD
- Status: Accepted
- Decision owners: DEV NEXUS
- Related task: CFG-001

Decision:

The team will not use feature branches. Collision risk will be controlled through task locks, single-file ownership, pull-rebase before edits and pushes, small commits, CI, and a ban on force pushes.

