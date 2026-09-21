# Model Card — Damaged-Part YOLOv8n v1 (5 Classes)

## Identity

- **Model version**: `DET-PART-001`
- **Task**: 5-class damaged vehicle component localisation
- **Classes (nc=5)**:
  - `0: headlamp`
  - `1: front_bumper`
  - `2: hood`
  - `3: door`
  - `4: rear_bumper`
- **Architecture**: Ultralytics YOLOv8n (`yolov8n.pt`, CSPDarknet53 backbone + PANet neck, ~3.2M parameters)
- **Base weights**: Pretrained on MS COCO (`yolov8n.pt`)
- **Experiment ID**: `DET-PART-001`
- **Training dataset**: COCO Car Damage Detection Dataset (`yolo_parts/` split converted in Phase 8)
- **Data splits**: 59 train (177 part boxes) / 11 val / 8 test images
- **Export format**: PyTorch `.pt` (`artifacts/models/parts_yolov8n.pt`) and ONNX (`artifacts/models/parts_yolov8n.onnx`)
- **Owner**: Member 4 (Detection ML) / Antigravity
- **Phase**: Phase 10
- **Companion models**: Phase 9 (`DET-YOLO-001`), Phase 11 (`INF-DEMO-001`), Phase 14 (`COST-RULE-001`)

---

## Intended Use

The model processes a single RGB vehicle photograph and localizes visible damaged automotive components (`headlamp`, `front_bumper`, `hood`, `door`, `rear_bumper`) with rectangular bounding boxes, component class labels, and confidence probabilities. In the ClaimVision AI triage pipeline:

1. **Downstream Costing Integration**: Component identifications directly feed the rule-based repair costing engine (Phase 14) to look up indicative INR replacement part costs and replacement labor rates.
2. **Reviewer Workspace Context**: Generates distinct color-coded component overlays (`DEFAULT_PART_COLORS`) in the human insurance adjuster review workspace (`predict_with_overlay`), enabling adjusters to quickly verify which components require repair or replacement.

---

## Prohibited Interpretation

> [!CAUTION]
> **This model outputs 2D rectangular bounding box coordinates locating visible damaged parts. It does NOT:**
> 1. Form a pixel-perfect segmentation mask (per `AGENTS.md` §10, bounding boxes must never be presented as masks).
> 2. Determine structural or unibody frame rail distortion concealed behind bumper fascias.
> 3. Guarantee that un-detected adjacent parts are undamaged.
> 4. Authorize insurance parts procurement or binding repair orders.

---

## Input / Output Contract

| Item | Specification |
|---|---|
| Input format | JPEG, PNG, or WEBP photograph |
| Input dimensions | Dynamic, letterboxed to $640 \times 640$ |
| Color channels | RGB / BGR (handled transparently) |
| Output type | Ordered list of `PartDetection` records |
| Bounding box (pixel) | `box_xyxy`: `[x1, y1, x2, y2]` in image pixel coordinates |
| Bounding box (norm) | `box_normalized`: `[cx, cy, w, h]` in $[0, 1]$ range |
| Confidence score | Float in range $[0.0, 1.0]$ |
| Class ID / Name | Integer $0 \dots 4$ matching `PARTS_CLASS_NAMES` |
| Color palette | Headlamp: Yellow, Front Bumper: Cyan, Hood: Green, Door: Blue, Rear Bumper: Magenta |
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

## Phase 10 Gate Decision: Conditional Production-Assistive Protocol

Per `README.md` §10 and §19, the Phase 10 gate requires an explicit operational decision:

1. **High Confidence ($\ge 0.40$)**:
   When the model predicts a damaged part with confidence $\ge 0.40$, the part name is forwarded to `claimvision_ml.costing` for line-item repair estimation.
2. **Ambiguous or Low Confidence ($< 0.40$)**:
   If part detection is ambiguous or returns no components, the system **gracefully falls back** to generic damage localisation (Phase 9) combined with overall image severity classification (Phases 5–7) to compute an indicative band without stalling the claim triage pipeline.
3. **Front vs. Rear Bumper Safeguard**:
   Due to geometric similarities on partial bumper corner crops, the adjuster UI flags bumper detections for quick human confirmation.

---

## Error Modes & Limitations

1. **Training Scale (177 Part Annotations)**: The dataset contains 177 part annotations across 59 training images. While sufficient for prototype triage demonstrations, rare vehicle models or complex aftermarket bodies may degrade accuracy.
2. **Close-Up Extreme Cropping**: When an image zooms into a crumpled bumper without showing lights or wheel wells, front-versus-rear bumper discrimination accuracy decreases.
3. **Multiple Damaged Components in Single Impact**: Heavily crumpled front clips where the bumper, hood, and headlamp fold into a single composite mass are demarcated with overlapping boxes.
