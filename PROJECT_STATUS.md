# Project Status

Last updated: 2026-09-21 01:45 IST
Updated by: Antigravity (CV-001)
Current commit: de7f653 (pending user commit for Phase 3)

## Overall state

| Field | Value |
| --- | --- |
| Current phase | Phase 3 Complete (Phase 4 Ready) |
| Overall health | Green |
| Next phase gate | Severity Dataset Audit — frozen 70/15/15 manifests |
| Demo readiness | Not started |
| Latest stable tag | None |
| Active blocker count | 0 |

## Phase tracker

| Phase | Owner | Status | Entry criteria | Exit evidence |
| --- | --- | --- | --- | --- |
| 0 Repository and controls | Member 1 | Complete | README approved | Setup verified (commit abf99fb) |
| 1 Fraud dataset audit | Member 2 | Complete | Data instructions ready | Audit notebook accepted (8,079 images audited) |
| 2 Fraud classifier | Member 2 | Complete | Frozen fraud manifests | FRAUD-MNV2-001 trained; test PR-AUC=0.5464; thresholds frozen; predict_fraud verified |
| 3 OpenCV evidence integrity | Member 3 | Complete | Sample images ready | runtime_checker.py + 19 tests pass (56 total); notebook 04 12 sections |
| 4 Severity audit | Member 3 | Not started | Dataset available | Frozen manifests |
| 5 Severity CNN | Member 3 | Not started | Severity audit accepted | Baseline report |
| 6 Severity MobileNetV2 | Member 3 | Not started | Same split available | Transfer-learning report |
| 7 ViT-Tiny and selection | Member 4 | Not started | Same split available | Selection decision |
| 8 COCO conversion | Member 4 | Not started | Detection annotations available | Visual conversion audit |
| 9 Damage YOLO | Member 4 | Not started | Conversion accepted | Model and metrics |
| 10 Part YOLO | Member 4 | Not started | Part labels verified | Model and metrics |
| 11 Unified inference | Member 5 | Not started | Selected models exported | Stable unified schema |
| 12 Backend foundation | Member 5 | Not started | API contract approved | Backend tests pass |
| 13 Assessment APIs | Member 5 | Not started | Inference adapter stable | Endpoint tests pass |
| 14 Customer UI | Member 6 | Not started | API mocks available | Main journey works |
| 15 Reviewer dashboard | Member 7 | Not started | Reviewer schema ready | Dashboard journey works |
| 16 Integration and testing | Members 1–7 | Not started | All core modules ready | Four scenarios pass 3 times |
| 17 Presentation freeze | Members 1–7 | Not started | Integration gate passed | Demo package frozen |
| 18 Core completion | Members 1–7 | Not started | Presentation approved | Core tagged and documented |

## Current sprint objectives

- [x] CFG-001 — Install repository configuration pack (done in commit 787c999)
- [x] CFG-002 — Create application skeleton (directory structure, ML package, backend placeholder, frontend placeholder, 16 notebook stubs)
- [x] Phase 0 gate — Backend health endpoint returns 200, ML package imports, frontend builds
- [x] DATA-001 — Fraud dataset audit (8,079 images; frozen manifests; 20 unit tests pass)
- [x] ML-001 — Fraud MobileNetV2 classifier (FRAUD-MNV2-001 trained; test PR-AUC 0.5464; high_threshold 0.80; recall 90.1%; ONNX export; 37 unit tests pass)
- [x] CV-001 — OpenCV evidence-integrity runtime checker (run_quality_checks(); QualityResult; 19 new tests; 56 total pass; notebook 04 12 sections)

## Active blockers

| ID | Blocker | Owner | Impact | Required decision | Target date |
| --- | --- | --- | --- | --- | --- |
| None | — | — | — | — | — |

## Latest verified results

Only paste results produced by committed code and recorded experiments.

| Module | Experiment/model version | Dataset version | Test result | Artifact |
| --- | --- | --- | --- | --- |
| Fraud (Audit) | Audit v1 (Vinay Jose) | 8,079 images (7,614 gen / 465 susp) | 0 leakage; 5,654 train, 1,211 val, 1,214 test | data/manifests/fraud_train.csv |
| Fraud (Classifier) | FRAUD-MNV2-001 | Vinay Jose v1 (1,214 test images) | PR-AUC: 0.5464, Recall: 90.1%, ROC-AUC: 0.9077, 22.6ms/img | ml/artifacts/fraud/thresholds_v1.json |
| Quality (Runtime) | CV-001 | Synthetic (tests) + real (notebook) | 19/19 tests pass; all 9 checks verified; EXIF rule confirmed | ml/src/claimvision_ml/quality/runtime_checker.py |
| Severity | Not available | Not available | Not measured | — |
| Damage detection | Not available | Not available | Not measured | — |
| Part detection | Not available | Not available | Not measured | — |
| End-to-end | Not available | Demo fixtures | Not tested | — |

## Next three actions

1. Review and commit Phase 3 — User — CV-001
2. Implement Phase 4 Severity Dataset Audit — Member 3 — SDATA-001
3. Create frozen 70/15/15 severity manifests — Member 3 — SDATA-001


## Demo readiness checklist

- [ ] Clean laptop setup verified
- [ ] All selected notebooks execute from top to bottom
- [ ] Model artifacts pass checksum validation
- [ ] Backend starts without retraining
- [ ] Frontend starts with documented command
- [ ] Four judge scenarios pass three consecutive times
- [ ] Offline fallback assets are available
- [ ] Seven member speaking roles rehearsed
