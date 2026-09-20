# Codex Single-Task Prompt

```text
Work on exactly one ClaimVision AI task: [TASK-ID — TITLE].

First read README.md, AGENTS.md, PROJECT_STATUS.md, TASKS.md, TASK_LOCKS.md,
docs/DECISIONS.md, and all files directly involved in the task.

Constraints:
- Current branch is main; do not create a branch.
- Base commit: [HASH].
- Allowed files: [PATHS].
- Do not touch: [PATHS].
- Owner/task lock: [NAME / LOCK ROW].
- Acceptance criteria: [PASTE].
- Required validation: [COMMANDS].

Inspect before editing. Preserve unrelated user changes. State the intended files,
then implement the smallest complete solution. Use apply-patch-style focused edits.
Never fabricate data, model results, test results, or documentation. For ML work,
keep judge-facing outputs in the required notebook and reusable logic in modules.
Run relevant validation and update project records. Do not force push, delete broad
paths, or resolve ambiguous scientific/product decisions without asking.

Finish with the completion report required by AGENTS.md and a suggested atomic
commit message. Do not claim a check passed unless you ran it successfully.
```

