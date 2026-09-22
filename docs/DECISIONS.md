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

### ADR-004 — CNN image classifier for damaged-part location

- Date: 2026-09-22
- Status: Accepted
- Decision owners: DEV NEXUS (mentor instruction)
- Related tasks: LOC-DATA-001, LOC-MNV2-001, LOC-EFF-001, LOC-COMP-001

Context:

The mentor directed adding a second type of part-level classification. YOLO (Notebooks 11–12) provides bounding-box localisation and per-box class labels. The mentor additionally requested an image-level CNN that answers "which vehicle part is predominantly damaged in this photo?" without any bounding box. This is analogous to the severity model comparison (Notebooks 06–09) but for location instead of damage severity.

The COCO Car Damage Detection Dataset (already used for YOLO) contains bounding-box annotations for five part classes: headlamp, front_bumper, hood, door, rear_bumper. No additional data was collected.

Options considered:

1. Multi-label classification (image can have multiple class labels). Requires more data and more complex threshold management. Rejected — dataset too small (~12 images per class) to reliably learn multi-label boundaries.
2. Single-label classification with dominant-part rule. Each image is assigned one label based on the most-annotated part class (area tiebreak). Simple, deterministic, reproducible. Selected.
3. Use YOLO class output only. Does not satisfy mentor requirement for an independent image-level CNN comparison.

Decision:

Add `ml/src/claimvision_ml/location/` subpackage with MobileNetV2 and EfficientNet-B0 image-level classifiers trained using 2-stage ImageNet transfer learning. Label derivation: dominant-part (most bounding boxes; area tiebreak). Compare both models in Notebook 15 and record the selected model. YOLO remains unchanged.

Consequences:

- Adds a scientifically defensible second approach to part identification for judge demonstration.
- Dataset constraint (~12 images per class) means results are prototype-quality only.
- Notebooks 13, 14, 15 added; former stubs 13–15 renumbered to 16–18.

Validation/revisit trigger:

If the judge demonstration shows <30% val accuracy on both models, disclose the dataset size as the root cause and present the models as "experimental" in the unified inference demo.
