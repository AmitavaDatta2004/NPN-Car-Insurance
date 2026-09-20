# ClaimVision AI Agent Operating Manual

This file is the binding operating contract for every human contributor, Antigravity agent, Codex session, and other coding assistant working in this repository.

## 1. Authority order

When instructions conflict, obey them in this order:

1. The human team leader's current written instruction.
2. This `AGENTS.md`.
3. The root `README.md` implementation plan.
4. Accepted architecture decisions in `docs/DECISIONS.md`.
5. The assigned task in `TASKS.md`.
6. Existing source code and comments.

Stop and ask the team leader if a conflict remains. Never silently choose a new architecture.

## 2. Project constraints

- Product: ClaimVision AI, an explainable vehicle-damage claim assessment prototype.
- Presentation target: local judge demonstration; production deployment is not required.
- Frontend: Next.js with TypeScript.
- Backend: FastAPI with Python.
- ML demonstrations: Jupyter notebooks with saved, reproducible judge-facing outputs.
- Reusable ML logic: Python modules imported by notebooks and backend.
- Implementation order: fraud first, severity second, localisation third, application integration fourth.
- Team: seven members.
- Git workflow: members commit directly to `main`; feature branches are not part of this workflow.
- Scientific rule: never manufacture data, metrics, model outputs, screenshots, or conclusions.

## 3. Mandatory reading before work

Before editing anything, read:

1. `README.md`.
2. This file.
3. `PROJECT_STATUS.md`.
4. `TASKS.md`.
5. `TASK_LOCKS.md`.
6. `docs/DECISIONS.md`.
7. Files directly related to the assigned task.

For an ML task, additionally read `docs/EXPERIMENT_LOG.md`, the applicable dataset card, model card, and notebook.

## 4. One-task execution protocol

Every agent must follow this sequence.

### Before editing

1. Confirm the task has an ID and acceptance criteria in `TASKS.md`.
2. Confirm prerequisites are complete.
3. Add or verify an active lock in `TASK_LOCKS.md`.
4. Pull the latest `main` using `git pull --rebase origin main`.
5. Run `git status --short`.
6. Never overwrite uncommitted work belonging to another member.
7. State the files that will be read, created, or changed.
8. State the validation commands that will be run.

### During editing

- Change only files required by the active task.
- Keep the patch small and reviewable.
- Do not combine unrelated formatting or refactoring.
- Preserve public API contracts unless the task explicitly changes them.
- Put reusable logic in modules; do not hide application logic inside notebooks.
- Update documentation in the same task when behavior, schemas, datasets, models, or commands change.
- Record irreversible or architecture-level decisions in `docs/DECISIONS.md`.
- Never place secrets, tokens, dataset archives, trained weights, or private data in Git.

### Before committing

1. Re-read the task acceptance criteria.
2. Run the smallest relevant tests plus formatting and static checks.
3. Inspect `git diff --check` and `git diff`.
4. Confirm no secret, dataset, model weight, cache, or notebook checkpoint is staged.
5. Update `PROJECT_STATUS.md`, `TASKS.md`, and applicable logs/cards.
6. Pull again with rebase immediately before pushing.
7. Resolve conflicts manually; never use force push.
8. Commit one logical change using the required format.
9. Push to `main`.
10. Remove the task lock only after the pushed commit is visible remotely.

## 5. Direct-to-main safety rules

Direct-to-main work is allowed only with these controls:

- One active editor per file.
- Every task must have a task lock before editing.
- A member must announce the task ID in the team channel.
- Pull-rebase before editing and again before pushing.
- Small atomic commits; do not hold large unpushed changes.
- Never use `git push --force`, `git reset --hard`, destructive checkout, or broad deletion.
- Never amend another member's published commit.
- If remote `main` changed in the same file, stop, coordinate, then rebase.
- If tests fail, do not push unless the failure is documented as pre-existing and the team leader approves.
- After every stable phase, create an annotated tag such as `phase-02-fraud-baseline`.

## 6. File ownership boundaries

The current owner is the person holding the task lock. Default areas are:

| Area | Default owner role |
| --- | --- |
| Fraud notebooks and fraud service | Fraud ML member |
| Severity notebooks and severity service | Severity ML member |
| Detection notebooks and detection service | Detection ML member |
| OpenCV and evidence integrity | Computer-vision member |
| FastAPI, database, orchestration | Backend member |
| Next.js customer flow | Frontend member |
| Reviewer dashboard, testing, demo | Integration/presentation member |

An owner may authorize another member in `TASK_LOCKS.md`. Ownership prevents collision; it does not prevent review.

## 7. Notebook contract

Every committed notebook must:

- open with title, purpose, owner, date, task ID, dataset version, and environment;
- declare the hypothesis or question being tested;
- set seeds and print important library versions;
- use repository-relative paths through the project configuration;
- show dataset counts and split integrity checks;
- preserve useful outputs required for the judge presentation;
- include plots with labels, units, legends, and readable sizes;
- show representative successes and failures;
- end with findings, limitations, artifact paths, and the next decision;
- restart and run all cells successfully before it is marked complete.

Notebooks must not:

- contain credentials or personal paths;
- download data on every ordinary run when it already exists;
- redefine large reusable classes already available in `ml/src`;
- display hundreds of images or huge tables;
- claim denoising, cropping, or enhancement improves performance without an ablation;
- claim image-level fraud classification proves real insurance fraud;
- use test data for tuning thresholds or choosing models.

### Notebook output policy

Keep outputs that help judges understand the pipeline: preprocessing grids, class distributions, learning curves, confusion matrices, PR/ROC curves, detections, Grad-CAM examples, latency tables, and selected failure cases. Clear noisy logs, progress-bar spam, secrets, and huge raw outputs.

## 8. Dataset contract

- Raw datasets are immutable.
- Store dataset archives and extracted images outside Git.
- Track only download instructions, manifests, schemas, checksums, licences, and small safe samples when permitted.
- Split by claim/vehicle/source group whenever related images can leak across splits.
- Generate and freeze train/validation/test manifests before model comparison.
- Never tune using the test set.
- Document class meaning, provenance, licensing, known biases, and exclusions in a dataset card.
- Verify every image can be decoded; quarantine corrupt files instead of silently deleting them.
- Record exact dataset version and checksum in each experiment.

## 9. ML experiment contract

Every training experiment receives a unique ID such as `FRAUD-MNV2-001` and records:

- Git commit;
- dataset and manifest version;
- model and pretrained weights;
- image size and augmentations;
- loss, optimizer, scheduler, epochs, batch size, and seed;
- hardware and duration;
- validation and untouched test metrics;
- threshold-selection method;
- artifact paths;
- failure notes and conclusion.

The selected model must be justified by accuracy, class-wise behavior, calibration, latency, size, and demo stability—not accuracy alone.

## 10. Model-specific rules

### Fraud

- Train fraud first.
- Treat the image model as a suspicious-image signal, not proof of legal fraud.
- Report PR-AUC, fraud precision/recall/F1, confusion matrix, calibration, and threshold behavior.
- Audit source, watermark, resolution, and background shortcuts.
- Missing EXIF alone must never classify an image as fraud.

### Severity

- Use the same frozen split for CNN, MobileNetV2, and ViT-Tiny.
- Report macro F1, weighted F1, class recall, confusion matrix, calibration, latency, and size.
- Keep severity labels ordinal in analysis even when trained as classification.
- Do not call crop predictions part-specific severity without matching labelled crop training.

### Detection and location

- Validate COCO-to-YOLO conversion visually before training.
- Separate generic damage localisation from damaged-part detection unless the dataset supports a combined taxonomy.
- Report mAP50, mAP50-95, precision, recall, per-class AP, latency, and qualitative failures.
- Never present a bounding box as a segmentation mask.

## 11. OpenCV rules

OpenCV provides deterministic evidence-quality checks, transformations, annotations, and visualisation. It is not the fraud or severity model.

For every transformation shown to judges:

- preserve the original image;
- display original and transformed images together;
- state the parameter values;
- explain the purpose;
- measure whether it improves the downstream task before enabling it by default.

Cropping, denoising, sharpening, contrast correction, and perspective correction are experiments until validated. All runtime operations must return warnings and measurements rather than silently altering evidence.

## 12. API contract

- All routes are versioned under `/api/v1`.
- Request and response bodies use Pydantic models.
- Errors follow one documented shape and never expose stack traces.
- Model loading occurs once during application startup or lazy singleton initialization.
- Uploads validate MIME type, extension, size, decode success, and image dimensions.
- Inference returns model version, threshold/version, confidence, warnings, and timing.
- High fraud risk routes to human review and prevents an automatic final recommendation.
- Use explicit state transitions; do not infer claim state from missing fields.
- API changes require OpenAPI review and frontend contract update.

## 13. Frontend contract

- Use TypeScript strict mode.
- Centralize API access and response types.
- Implement loading, empty, error, retry, and success states.
- Display uncertainty and warnings; never imply the AI decision is final.
- Retain an original-image view beside annotated evidence.
- Ensure the demo works at common laptop resolution without horizontal overflow.
- Use accessible labels, keyboard navigation, visible focus, alt text, and sufficient contrast.
- Do not hard-code fake model metrics as if they were live results.

## 14. Testing contract

Minimum tests for a changed module:

- Python unit tests for pure logic.
- API integration tests for changed endpoints.
- Frontend component or flow tests for changed UI behavior.
- ML smoke inference test for each exported model.
- Schema compatibility test for the unified response.
- A manual end-to-end run for the four judge scenarios before presentation.

No task is complete merely because the application starts.

## 15. Commands and environment

Use commands documented in the root `README.md`. If a command is missing or wrong, update the README as part of the task. Use `.env.example` as the schema; never edit or commit another member's `.env`.

Prefer deterministic commands. Do not upgrade dependencies opportunistically. Any version change must state why and update the lockfile.

## 16. Required completion report

Every agent must finish with:

```text
Task ID:
Outcome:
Files changed:
Commands run:
Tests and results:
Generated artifacts:
Documentation updated:
Known limitations:
Follow-up task:
Commit hash:
```

## 17. Mandatory stop conditions

Stop and ask the team leader when:

- the task lacks acceptance criteria;
- required data or labels do not exist;
- dataset licensing is unclear;
- a schema conflicts with the README or accepted decision;
- another person holds a lock on a required file;
- uncommitted unfamiliar changes overlap the task;
- a model result contradicts the proposed product claim;
- a destructive operation appears necessary;
- credentials or private claim data would be exposed;
- the requested work expands beyond the active phase.

## 18. Prohibited agent behavior

Agents must never:

- fabricate completion, metrics, datasets, or screenshots;
- silently change scope;
- create parallel alternative architectures without approval;
- replace working code wholesale for convenience;
- commit secrets, local absolute paths, raw datasets, weights, or caches;
- bypass tests or remove a failing test merely to obtain green CI;
- run destructive Git commands;
- push forcefully;
- modify another active member's files;
- represent prototype rules as insurer-approved decisions.

