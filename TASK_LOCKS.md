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
| CFG-002 | Antigravity | backend/, frontend/, ml/, notebooks/, data/, artifacts/, scripts/, annotated/, reports/, demo/, docs/agent-work-log.md, CHANGELOG.md, CONTRIBUTING.md, config/project.yaml, .github/workflows/ci.yml, PROJECT_STATUS.md, TASKS.md | 2026-09-20 14:16 | 2026-09-20 17:00 | 787c999 | ACTIVE |

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

