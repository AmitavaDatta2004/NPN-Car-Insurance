# Judge Demonstration Runbook

## Presentation freeze

- Freeze code and model artifacts at least one day before judging.
- Record the commit and create an annotated presentation tag.
- Do not retrain, upgrade dependencies, or redesign screens after freeze unless fixing a demonstrated blocker.
- Keep a copy of required models and demo images on every demo laptop.

## Pre-demo verification

1. Confirm power, charger, display adapter, and offline files.
2. Disable disruptive updates and notifications.
3. Verify Python, Node, and environment files.
4. Verify model checksums.
5. Start FastAPI and call health/model-readiness endpoints.
6. Start Next.js and complete the four flows.
7. Open judge-facing notebooks at the required output cells.
8. Confirm no private paths or credentials are visible.
9. Run all scenarios three consecutive times.

## Four demonstration scenarios

### Scenario 1 — Valid minor damage

Expected: evidence passes quality checks, fraud risk remains low, minor severity and localisation appear, cost range is explained, reviewer can inspect original and annotation.

### Scenario 2 — Valid major damage

Expected: major severity and relevant locations appear, recommendation is review-oriented, cost range and uncertainty are visible.

### Scenario 3 — Suspicious or duplicate evidence

Expected: fraud/evidence signal triggers human review, downstream automatic finalisation is stopped, reason codes are shown.

### Scenario 4 — Poor-quality image

Expected: blur/darkness/glare or decode issue is reported, user receives capture guidance, no confident final claim decision is fabricated.

## Speaking sequence for seven members

| Member | Topic | Target time |
| --- | --- | --- |
| 1 | Problem, architecture, and workflow | 60–90 seconds |
| 2 | Fraud data, model, metrics, and limitations | 60–90 seconds |
| 3 | OpenCV and severity baseline/transfer learning | 60–90 seconds |
| 4 | ViT comparison and YOLO localisation | 60–90 seconds |
| 5 | FastAPI, orchestration, decision and cost engines | 60–90 seconds |
| 6 | Customer Next.js journey | 60–90 seconds |
| 7 | Reviewer dashboard, testing, impact, and next phases | 60–90 seconds |

## Fallback plan

- If live inference fails, show previously generated outputs explicitly labelled as recorded results.
- If the backend fails, use a documented local mock only for UI demonstration and say so.
- If one notebook is slow, show saved verified output and the corresponding experiment record.
- Never claim a fallback result was generated live.

## Post-demo evidence

- Final commit and tag
- Notebook filenames
- Model cards and experiment IDs
- Test report
- Known limitations
- Advanced phases already implemented versus still pending

