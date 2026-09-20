# Antigravity Single-Task Prompt

Copy this prompt and replace every placeholder. Assign only one task per run.

```text
You are working on ClaimVision AI.

Mandatory reading, in order:
1. README.md
2. AGENTS.md
3. PROJECT_STATUS.md
4. TASKS.md
5. TASK_LOCKS.md
6. docs/DECISIONS.md
7. Files related to TASK_ID

Execute only TASK_ID: [TASK-ID — TITLE].
Owner: [NAME]
Current phase: [PHASE]
Base commit: [HASH]
Allowed files: [EXACT PATHS]
Prohibited files: [EXACT PATHS OR AREAS]
Inputs: [INPUTS]
Expected outputs: [OUTPUTS]
Acceptance criteria: [PASTE FROM TASKS.md]
Validation commands: [COMMANDS]

Before editing:
- Check git status and the active task locks.
- Stop if another member owns any required file.
- Read the existing implementation before proposing changes.
- Reply with a short understanding, exact file plan, and validation plan.
- Do not edit until the scope is internally consistent.

During work:
- Follow AGENTS.md exactly.
- Keep the patch limited to this task.
- Reuse existing architecture and utilities.
- Never invent datasets, labels, metrics, API fields, successful outputs, or completed tests.
- Put reusable ML logic in Python modules and judge-facing evidence in the named notebook.
- Update task/status/experiment/model/dataset documentation when applicable.

Before completion:
- Run the specified tests and relevant lint/type checks.
- Inspect the complete diff.
- Do not push if validation fails.
- Do not force push or use destructive Git commands.

Return exactly:
Task ID:
Outcome:
Files changed:
Commands run:
Tests and results:
Generated artifacts:
Documentation updated:
Known limitations:
Follow-up task:
Suggested commit message:
```

