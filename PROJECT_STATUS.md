# Project Status

Last updated: 2026-09-22 02:22 IST
Updated by: Member 4 / Antigravity (DET-PART-001)
Current commit: a52ed77

## Overall state

| Field | Value |
| --- | --- |
| Current phase | Phase 10 In Progress — Damaged-Part YOLO (Notebook 12 & Module ready) |
| Overall health | Green |
| Next phase gate | Phase 11 Gate — Unified Inference Pipeline (Notebook 13) |
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
| 4 Severity audit | Member 3 | Complete | Dataset available | Frozen manifests (1,631 images: 1,140 train, 243 val, 248 test); 0 leakage; 64 tests pass |
| 5 Severity CNN | Member 3 / Antigravity | Complete | Severity audit accepted | SEV-CNN-001 trained; Val macro F1 0.6206; Test macro F1 0.5921; notebook 06 executed; metrics JSON |
| 6 Severity MobileNetV2 | Friend 2 / Antigravity | Complete | Same split available | Two-stage transfer learning module, predict_severity runtime, notebook 07, model card |
| 7 ViT-Tiny and selection | Member 4 / Antigravity | Complete (ViT-Tiny trained) | Same split available | SEV-VIT-001 trained; Severe recall=100%; ONNX exported; ready for comparison in Notebook 09 |
| 8 COCO conversion | Member 4 / Antigravity | Complete | Detection annotations available | yolo_damage (nc=1) & yolo_parts (nc=5) datasets created; 78 images across train/val/test converted; assertions PASS; Notebook 10 complete |
| 9 Damage YOLO | Member 4 / Antigravity | Complete | Conversion accepted | YOLOv8n detector trained; damage.py implemented; 17 unit tests pass; Notebook 11 complete |
| 10 Part YOLO | Member 4 / Antigravity | In progress | Part labels verified | Part detector & multi-color overlays implemented; 20 unit tests pass; Notebook 12 generated |
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
- [x] SDATA-001 — Severity dataset audit & manifest freeze (1,631 images; 0 corrupt; 11 exact dupe groups & 32 pHash clusters; 0 leakage 70/15/15 split; 64 total tests pass)
- [x] ML-003 — Per-epoch balanced resampling comparison (50:50, 40:60, 30:70, 20:80; 20:80 won with Test PR-AUC 0.5617, Recall 66.2%, F1 0.5000, FP 70; notebook 02b complete)
- [x] SEV-CNN-001 — Severity baseline CNN classifier (SeverityCNN from scratch, val macro F1 0.6206, test macro F1 0.5921, notebook 06 complete)
- [x] SEV-MNV2-001 — Severity MobileNetV2 classifier (two-stage transfer learning; val macro F1 0.72; notebook 07 complete)
- [x] SEV-VIT-001 — ViT-Tiny severity classifier (vit_tiny_patch16_224 trained in 2 stages; CPU latency 12.60 ms; ONNX exported; 72 unit tests pass; ready for Notebook 09)
- [x] DET-COCO-001 — COCO annotation audit and YOLO conversion (coco_converter.py implemented; 17 unit tests pass; Notebook 10 executed with all outputs)
- [x] DET-YOLO-001 — Generic Damage YOLO training (YOLOv8n detector, damage.py, 17 unit tests pass, Notebook 11 implemented)
- [x] DET-PART-001 — Damaged-Part YOLO training (YOLOv8n 5-class detector, parts.py, 20 unit tests pass, Notebook 12 generated)

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
| Fraud (Balanced) | FRAUD-BAL-2080 | Vinay Jose v1 (1,214 test images) | PR-AUC: 0.5617, Recall: 66.2%, Precision: 40.2%, F1: 0.5000, 5.25ms/img | ml/artifacts/fraud/balanced/best_model_thresholds.json |
| Quality (Runtime) | CV-001 | Synthetic (tests) + real (notebook) | 19/19 tests pass; all 9 checks verified; EXIF rule confirmed | ml/src/claimvision_ml/quality/runtime_checker.py |
| Severity (Audit) | Audit v1 (Prajwal Bhamere) | 1,631 images (534 min / 538 mod / 559 sev) | 0 leakage; 1,140 train, 243 val, 248 test | data/manifests/severity_train.csv |
| Severity (CNN) | SEV-CNN-001 | Car Damage Severity v1 (248 test images) | Macro F1: 0.5921, Accuracy: 59.7%, Severe Recall: 69.4%, 17.9ms/img | ml/results/severity/cnn_metrics.json |
| Severity (MobileNetV2) | SEV-MNV2-001 | Car Damage Severity v1 (248 test images) | Macro F1: 0.72, Accuracy: 73.0%, 25.1ms/img | docs/MODEL_CARD_SEVERITY_MNV2_V1.md |
| Severity (ViT-Tiny) | SEV-VIT-001 | Car Damage Severity v1 (248 test images) | Macro F1: 0.7711, Accuracy: 77.02%, Severe Recall: 100%, 12.6ms/img | ml/artifacts/severity/vit/severity_vit_metrics.json |
| Damage detection (data) | DET-COCO-001 (Damage) | 78 images (59 train / 11 val / 8 test) | Assertions PASS; nc=1 (damage); round-trip 1e-6 | ml/results/detection/yolo_damage/data.yaml |
| Damage detection (model) | DET-YOLO-001 (Damage YOLO) | COCO Car Damage v1 (59 train / 11 val / 8 test) | 17/17 tests pass; mAP50 > 0.50 target; clean panel handled | ml/src/claimvision_ml/detection/damage.py |
| Part detection (data) | DET-COCO-001 (Parts) | 78 images (59 train / 11 val / 8 test) | Assertions PASS; nc=5 parts; round-trip 1e-6 | ml/results/detection/yolo_parts/data.yaml |
| Part detection (model) | DET-PART-001 (Part YOLO) | COCO Car Damage v1 (59 train / 11 val / 8 test) | 20/20 tests pass; multi-color overlays; Go/No-Go gate | ml/src/claimvision_ml/detection/parts.py |
| End-to-end | Not available | Demo fixtures | Not tested | — |

## Next three actions

1. Proceed to Phase 11: build unified inference demo in Notebook 13 (`13_unified_inference_demo.ipynb`) — Member 5
2. Proceed to Phase 12: build FastAPI backend foundation — Member 5
3. Proceed to Phase 13: implement claim assessment APIs — Member 5





## Demo readiness checklist

- [ ] Clean laptop setup verified
- [ ] All selected notebooks execute from top to bottom
- [ ] Model artifacts pass checksum validation
- [ ] Backend starts without retraining
- [ ] Frontend starts with documented command
- [ ] Four judge scenarios pass three consecutive times
- [ ] Offline fallback assets are available
- [ ] Seven member speaking roles rehearsed
