# Dataset Card — Location Classification Dataset (COCO Car Damage, 5 Part Classes)

**Task IDs:** LOC-DATA-001, LOC-MNV2-001, LOC-EFF-001  
**Created:** 2026-09-22  
**Owner:** Location CNN member / Antigravity  
**Version:** v1 (same physical dataset as DET-COCO-001)

---

## Dataset Overview

This card describes the use of the **COCO Car Damage Detection Dataset** as a source for the location classification task. The raw dataset is identical to the one used for YOLO training (Phases 9–10). No new images were collected.

| Field | Value |
|---|---|
| Source dataset | COCO Car Damage Detection Dataset (Kaggle) |
| Original task | Object detection (bounding boxes) |
| Derived task | Image-level location classification |
| Number of images | 59 train / 11 val / 8 test (unannotated) |
| Part classes | 5: headlamp, front_bumper, hood, door, rear_bumper |
| Annotation format | COCO JSON (bounding boxes) |
| Licence | See original Kaggle dataset page |

---

## Label Derivation Strategy

### Why derived labels?

The COCO dataset provides **bounding-box annotations** (multiple boxes per image, multiple classes per image). The location CNN task requires **one label per image** (single-label classification). A label derivation algorithm converts multi-box annotations to single labels.

### Dominant-Part Rule

For each image:

1. Count the number of bounding boxes belonging to each of the 5 part classes.
2. The class with the **highest count** is assigned as the label.
3. On a tie (equal bounding-box count), the class with the **greatest total bounding-box area** is selected.

This rule is:
- Deterministic and reproducible.
- Consistent with human visual intuition (the most-annotated part is likely the focus of damage in the image).
- Implemented in `ml/src/claimvision_ml/location/dataset.py` → `derive_location_labels()`.

### Limitations of derived labels

- An image with headlamp + door annotations could plausibly be labelled as either class.
- The dominant-part rule does not reflect which part is most **severely** damaged.
- Derived labels are not manually verified — they are a best-effort proxy.

---

## Class Distribution (Estimated from COCO annotations)

> **Note:** Exact per-class counts depend on how many images have each part as the dominant annotated part. Run `derive_location_labels(COCO_mul_train_annos.json)` and count to reproduce.

| Class | Approx. Train Images | Approx. Val Images |
|---|---|---|
| headlamp | ~12 | ~2 |
| front_bumper | ~12 | ~2 |
| hood | ~12 | ~2 |
| door | ~12 | ~2 |
| rear_bumper | ~11 | ~3 |
| **Total** | **~59** | **~11** |

> These are approximations. Actual counts depend on annotation density per image.

---

## Critical Constraints

> [!CAUTION]
> **~12 training images per class.** This is an extremely small dataset for deep learning classification. Both MobileNetV2 and EfficientNet-B0 use ImageNet pretrained weights and 2-stage fine-tuning to compensate. Results are **prototype-quality** and must not be presented as production-grade accuracy.

- No manual label verification was performed after derivation.
- The test split (8 images) is **unannotated** — only visual qualitative inspection is possible.
- Some images may show damage to multiple parts simultaneously; the derived label picks only the dominant one.

---

## Data Splits

| Split | Images | Labels available | Use |
|---|---|---|---|
| Train | 59 | Yes (from COCO_mul_train_annos.json) | Model training |
| Val | 11 | Yes (from COCO_mul_val_annos.json) | Threshold selection, model comparison |
| Test | 8 | No (unannotated) | Qualitative visual inspection only |

The same physical image files and COCO JSON files are used as in Phase 8 (YOLO conversion). No re-splitting or re-downloading is needed.

---

## Category Name Normalisation

The COCO JSON may contain category names with spaces (e.g., `"rear bumper"`, `"front bumper"`). The `derive_location_labels()` function normalises these to underscore form:

| COCO JSON name | Canonical class name |
|---|---|
| `"headlamp"` | `headlamp` |
| `"front_bumper"` | `front_bumper` |
| `"front bumper"` | `front_bumper` |
| `"hood"` | `hood` |
| `"door"` | `door` |
| `"rear_bumper"` | `rear_bumper` |
| `"rear bumper"` | `rear_bumper` |

---

## Known Biases

1. **Camera angle bias:** most COCO images are taken from standard angles. Edge-case perspectives (aerial, very oblique) may not be represented.
2. **Make and model bias:** the dataset likely covers a limited range of vehicle makes and colours.
3. **Damage severity correlation:** front/rear bumper images may correlate with higher-severity (impact) damage, which could create spurious correlations with the severity classifier.

---

## How to reproduce label derivation

```python
from claimvision_ml.location.dataset import derive_location_labels
labels = derive_location_labels("path/to/COCO_mul_train_annos.json")
# Returns {filename: (class_id, class_name)}
```

---

## Related files

| File | Purpose |
|---|---|
| `ml/src/claimvision_ml/location/dataset.py` | `derive_location_labels`, `LocationDataset` |
| `ml/tests/test_location_dataset.py` | Unit tests for label derivation |
| `data/raw/coco_car_damage/` | Raw COCO images and annotation JSONs |
| `docs/DATASET_CARD_SEVERITY.md` | Severity dataset card (reference format) |
| `docs/DECISIONS.md` ADR-004 | Architecture decision for this label strategy |
