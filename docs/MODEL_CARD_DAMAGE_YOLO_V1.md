# Model Card — Generic Damage YOLOv8n v1

## Identity

ML-IMPROVE-001 clarification (2026-09-22): performance targets below are not measured achievements. Notebook 11b provides isolated validation-only YOLOv8n augmentation comparisons for generic damage and damaged parts. The supplied eight test images are unannotated and cannot support test mAP/precision/recall. Only audited annotated train/validation splits are used; new results remain pending Colab training. Existing Notebook 11 is preserved because remote main received a teammate update during this task.

- **Model version**: `DET-YOLO-001`
- **Task**: Single-class generic exterior vehicle damage localisation (`damage`, `nc=1`)
- **Architecture**: Ultralytics YOLOv8n (`yolov8n.pt`, CSPDarknet53 backbone + PANet neck, ~3.2M parameters)
- **Base weights**: Pretrained on MS COCO (`yolov8n.pt`)
- **Experiment ID**: `DET-YOLO-001`
- **Training dataset**: COCO Car Damage Detection Dataset (`yolo_damage/` split converted in Phase 8)
- **Data splits**: 59 train / 11 val / 8 test images
- **Export format**: PyTorch `.pt` (`artifacts/models/damage_yolov8n.pt`) and ONNX (`artifacts/models/damage_yolov8n.onnx`)
- **Owner**: Member 4 (Detection ML) / Antigravity
- **Phase**: Phase 9
- **Companion models**: Phase 8 (`DET-COCO-001`), Phase 10 (`DET-PART-001`), Phase 11 (`INF-DEMO-001`)

---

## Intended Use

The model processes a single RGB vehicle photograph and localizes visible collision damage, dents, scratches, and structural crumpling with rectangular bounding boxes and associated confidence scores. In the ClaimVision AI triage pipeline:

1. **Pipeline Ordering**: Executes after OpenCV quality checks (Phase 3) and fraud risk screening (Phases 1–2).
2. **Evidence Visualisation**: Feeds annotated bounding-box overlays to the human insurance adjuster review workspace (`predict_with_overlay`).
3. **Downstream Context**: Provides damaged region context alongside the image-level severity classification (Phases 5–7) and rule-based repair costing engine.

---

## Prohibited Interpretation

> [!CAUTION]
> **This model outputs 2D rectangular bounding box coordinates locating visible exterior damage. It does NOT:**
> 1. Form a segmentation mask (per `AGENTS.md` §10, bounding boxes must never be presented as pixel masks).
> 2. Inspect internal mechanical, engine, frame, or electronic sensor damage concealed behind body panels.
> 3. Authorize or finalize insurance repair payouts or legally binding claim decisions.
> 4. Determine driver fault or accident causality.

---

## Input / Output Contract

| Item | Specification |
|---|---|
| Input format | JPEG, PNG, or WEBP photograph |
| Input dimensions | Dynamic, letterboxed to $640 \times 640$ |
| Color channels | RGB / BGR (handled transparently by OpenCV and Ultralytics) |
| Output type | Ordered list of `DamageDetection` records |
| Bounding box (pixel) | `box_xyxy`: `[x1, y1, x2, y2]` in image pixel coordinates |
| Bounding box (norm) | `box_normalized`: `[cx, cy, w, h]` in $[0, 1]$ range |
| Confidence score | Float in range $[0.0, 1.0]$ |
| Class ID / Name | `class_id: 0`, `class_name: "damage"` |
| No-detection behavior | Returns empty list `[]` without error or exception |

---

## Training Configuration

| Setting | Value |
|---|---|
| Framework | Ultralytics YOLOv8 (v8.4+) |
| Optimizer | AdamW |
| Initial Learning Rate (`lr0`) | `0.001` |
| Image size (`imgsz`) | `640` |
| Batch size | `16` (CUDA GPU) / `8` (CPU) |
| Maximum epochs | `50` |
| Early stopping patience | `15` epochs |
| Seed | `42` |
| Checkpoint selection | Best Validation mAP50 (`weights/best.pt`) |

---

## Performance Targets & Safety Gates

| Metric | Target | Rationale |
|---|---|---|
| Validation mAP50 | $> 0.50$ | Minimum localization fidelity on visible exterior damage |
| Validation Recall | $> 0.60$ | Minimize missed damage in claims |
| CPU Inference Latency | $< 30$ ms/image | Seamless real-time execution on hackathon presentation laptop |
| Model File Size | $< 15$ MB | Ultra-compact deployment footprint |
| Clean Image Edge Case | $0$ false-alarm crashes | Clean vehicles must return empty detections without raising unhandled errors |

---

## Error Modes & Limitations

1. **Small Training Set (59 Images)**: The public COCO car damage dataset provides 59 annotated training images. While transfer learning from COCO provides robust initial features, the model is strictly a **hackathon proof-of-concept prototype** and must not be characterized as insurer-production ready.
2. **Specular Reflections**: High-contrast sun reflections and streetlamp glares on clean curved metallic panels can occasionally trigger low-confidence false positives. Mitigated via a conservative `conf_threshold = 0.25`.
3. **Hairline Scratches**: At 640px resolution, micro-abrasions and thin paint keying may fall below detection thresholds. Adjusters rely on original zoom in the ClaimVision workspace.

---

## Ethical & Transparency Considerations

- Human-in-the-loop: Bounding boxes are displayed alongside pristine original photographs in the reviewer UI. Adjusters can add, remove, or modify detected bounding boxes before approving the assessment report.
