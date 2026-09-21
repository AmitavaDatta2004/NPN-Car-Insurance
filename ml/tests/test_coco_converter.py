"""test_coco_converter.py — Unit tests for the Phase 8 COCO → YOLO converter.

Task ID : DET-COCO-001
Phase   : 8
Owner   : Member 4 / Antigravity
Run     : .venv\\Scripts\\pytest.exe ml/tests/test_coco_converter.py -v
"""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import numpy as np
import pytest

from claimvision_ml.detection.coco_converter import COCOtoYOLOConverter, ValidationResult


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_coco(
    *,
    n_images: int = 2,
    img_w: int = 640,
    img_h: int = 480,
    with_annotations: bool = True,
    category_ids: list[int] | None = None,
) -> dict:
    """Build a minimal synthetic COCO dict for testing."""
    if category_ids is None:
        category_ids = [1]

    images = [
        {"id": i + 1, "file_name": f"img_{i+1:03d}.jpg", "width": img_w, "height": img_h}
        for i in range(n_images)
    ]
    categories = [{"id": cid, "name": f"class_{cid}"} for cid in category_ids]
    annotations = []
    if with_annotations:
        for i, img in enumerate(images):
            annotations.append(
                {
                    "id": i + 1,
                    "image_id": img["id"],
                    "category_id": category_ids[0],
                    # A valid box: 100x80 at (50, 60)
                    "bbox": [50.0, 60.0, 100.0, 80.0],
                }
            )
    return {"images": images, "annotations": annotations, "categories": categories}


@pytest.fixture
def tmp_image_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with dummy image files."""
    for i in range(1, 4):
        (tmp_path / f"img_{i:03d}.jpg").write_bytes(b"FAKE")
    return tmp_path


@pytest.fixture
def simple_coco() -> dict:
    return _make_coco(n_images=2, img_w=640, img_h=480, with_annotations=True, category_ids=[1])


# ---------------------------------------------------------------------------
# Test 1 — convert_bbox_to_yolo: known values
# ---------------------------------------------------------------------------


def test_convert_bbox_to_yolo_known_values():
    """cx/cy/w/h should match hand-calculated normalised values."""
    cx, cy, w, h = COCOtoYOLOConverter.convert_bbox_to_yolo(
        x=50, y=60, w=100, h=80, img_w=640, img_h=480
    )
    assert math.isclose(cx, (50 + 50) / 640, rel_tol=1e-9)
    assert math.isclose(cy, (60 + 40) / 480, rel_tol=1e-9)
    assert math.isclose(w, 100 / 640, rel_tol=1e-9)
    assert math.isclose(h, 80 / 480, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Test 2 — convert_bbox_to_yolo: round-trip within 1e-6
# ---------------------------------------------------------------------------


def test_convert_bbox_to_yolo_round_trip():
    """Back-converting YOLO coords to pixels must match originals within 1e-6."""
    orig_x, orig_y, orig_w, orig_h = 123.0, 77.5, 200.0, 150.0
    img_w, img_h = 1280, 720

    cx, cy, nw, nh = COCOtoYOLOConverter.convert_bbox_to_yolo(
        orig_x, orig_y, orig_w, orig_h, img_w, img_h
    )

    rec_x = cx * img_w - nw * img_w / 2
    rec_y = cy * img_h - nh * img_h / 2
    rec_w = nw * img_w
    rec_h = nh * img_h

    assert abs(rec_x - orig_x) < 1e-6 + 1e-6 * abs(orig_x)
    assert abs(rec_y - orig_y) < 1e-6 + 1e-6 * abs(orig_y)
    assert abs(rec_w - orig_w) < 1e-6 + 1e-6 * abs(orig_w)
    assert abs(rec_h - orig_h) < 1e-6 + 1e-6 * abs(orig_h)


# ---------------------------------------------------------------------------
# Test 3 — convert_bbox_to_yolo: zero-width raises ValueError
# ---------------------------------------------------------------------------


def test_convert_bbox_to_yolo_zero_width_raises():
    with pytest.raises(ValueError, match="width"):
        COCOtoYOLOConverter.convert_bbox_to_yolo(10, 10, 0, 50, 640, 480)


# ---------------------------------------------------------------------------
# Test 4 — convert_bbox_to_yolo: zero-height raises ValueError
# ---------------------------------------------------------------------------


def test_convert_bbox_to_yolo_zero_height_raises():
    with pytest.raises(ValueError, match="height"):
        COCOtoYOLOConverter.convert_bbox_to_yolo(10, 10, 50, 0, 640, 480)


# ---------------------------------------------------------------------------
# Test 5 — validate_structure: passes on a valid synthetic COCO dict
# ---------------------------------------------------------------------------


def test_validate_structure_valid(simple_coco, tmp_image_dir):
    # Write the expected dummy files
    for img in simple_coco["images"]:
        (tmp_image_dir / img["file_name"]).write_bytes(b"FAKE")

    result = COCOtoYOLOConverter.validate_structure(simple_coco, tmp_image_dir)
    assert isinstance(result, ValidationResult)
    assert result.is_valid
    assert result.image_count == 2
    assert result.annotation_count == 2
    assert result.missing_files == []
    assert result.invalid_boxes == []


# ---------------------------------------------------------------------------
# Test 6 — validate_structure: fails when 'annotations' key missing
# ---------------------------------------------------------------------------


def test_validate_structure_missing_annotations_key(tmp_image_dir):
    bad_coco = {"images": [], "categories": []}
    result = COCOtoYOLOConverter.validate_structure(bad_coco, tmp_image_dir)
    assert not result.is_valid
    assert any("annotations" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Test 7 — validate_structure: flags bbox exceeding image width
# ---------------------------------------------------------------------------


def test_validate_structure_bbox_exceeds_width(tmp_image_dir):
    coco = _make_coco(n_images=1, img_w=100, img_h=200, with_annotations=False)
    coco["annotations"] = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [50, 10, 80, 50]}  # x+w=130 > 100
    ]
    (tmp_image_dir / "img_001.jpg").write_bytes(b"FAKE")
    result = COCOtoYOLOConverter.validate_structure(coco, tmp_image_dir)
    assert len(result.invalid_boxes) > 0
    assert any("width" in b["reason"] for b in result.invalid_boxes)


# ---------------------------------------------------------------------------
# Test 8 — validate_structure: flags bbox exceeding image height
# ---------------------------------------------------------------------------


def test_validate_structure_bbox_exceeds_height(tmp_image_dir):
    coco = _make_coco(n_images=1, img_w=200, img_h=100, with_annotations=False)
    coco["annotations"] = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 60, 50, 80]}  # y+h=140 > 100
    ]
    (tmp_image_dir / "img_001.jpg").write_bytes(b"FAKE")
    result = COCOtoYOLOConverter.validate_structure(coco, tmp_image_dir)
    assert len(result.invalid_boxes) > 0
    assert any("height" in b["reason"] for b in result.invalid_boxes)


# ---------------------------------------------------------------------------
# Test 9 — validate_structure: counts annotations per category correctly
# ---------------------------------------------------------------------------


def test_validate_structure_category_counts(tmp_image_dir):
    coco = _make_coco(n_images=3, img_w=640, img_h=480, with_annotations=False, category_ids=[1, 2])
    coco["annotations"] = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50]},
        {"id": 2, "image_id": 2, "category_id": 1, "bbox": [10, 10, 50, 50]},
        {"id": 3, "image_id": 3, "category_id": 2, "bbox": [10, 10, 50, 50]},
    ]
    for img in coco["images"]:
        (tmp_image_dir / img["file_name"]).write_bytes(b"FAKE")
    result = COCOtoYOLOConverter.validate_structure(coco, tmp_image_dir)
    assert result.category_counts["class_1"] == 2
    assert result.category_counts["class_2"] == 1


# ---------------------------------------------------------------------------
# Test 10 — draw_coco_boxes: returns same-size image
# ---------------------------------------------------------------------------


def test_draw_coco_boxes_preserves_size():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    annotations = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [50, 60, 100, 80]}]
    categories = {1: "damage"}
    out = COCOtoYOLOConverter.draw_coco_boxes(img, annotations, categories)
    assert out.shape == img.shape
    # Original must be unchanged
    assert np.all(img == 0)


# ---------------------------------------------------------------------------
# Test 11 — convert_split: correct number of .txt label files
# ---------------------------------------------------------------------------


def test_convert_split_label_file_count(tmp_path):
    coco = _make_coco(n_images=3, img_w=640, img_h=480, with_annotations=True, category_ids=[1])
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for img in coco["images"]:
        (img_dir / img["file_name"]).write_bytes(b"FAKE")

    out_dir = tmp_path / "yolo_out"
    category_map = {1: 0}
    COCOtoYOLOConverter.convert_split(coco, img_dir, out_dir, category_map, ["damage"])

    label_files = list((out_dir / "labels").glob("*.txt"))
    assert len(label_files) == 3, f"Expected 3 label files, got {len(label_files)}"


# ---------------------------------------------------------------------------
# Test 12 — convert_split: empty .txt for images with no annotations
# ---------------------------------------------------------------------------


def test_convert_split_empty_label_for_unannotated(tmp_path):
    coco = _make_coco(n_images=2, img_w=640, img_h=480, with_annotations=False, category_ids=[1])
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for img in coco["images"]:
        (img_dir / img["file_name"]).write_bytes(b"FAKE")

    out_dir = tmp_path / "yolo_out"
    COCOtoYOLOConverter.convert_split(coco, img_dir, out_dir, {1: 0}, ["damage"])

    for txt in (out_dir / "labels").glob("*.txt"):
        assert txt.read_text().strip() == "", f"{txt.name} should be empty but has content"


# ---------------------------------------------------------------------------
# Test 13 — convert_split: all label values within [0, 1]
# ---------------------------------------------------------------------------


def test_convert_split_values_in_range(tmp_path):
    coco = _make_coco(n_images=2, img_w=640, img_h=480, with_annotations=True, category_ids=[1])
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for img in coco["images"]:
        (img_dir / img["file_name"]).write_bytes(b"FAKE")

    out_dir = tmp_path / "yolo_out"
    COCOtoYOLOConverter.convert_split(coco, img_dir, out_dir, {1: 0}, ["damage"])

    for txt in (out_dir / "labels").glob("*.txt"):
        for line in txt.read_text().splitlines():
            if line.strip():
                parts = line.split()
                for val in parts[1:]:
                    assert 0.0 <= float(val) <= 1.0, f"Value {val} out of range in {txt.name}"


# ---------------------------------------------------------------------------
# Test 14 — write_data_yaml: correct nc and names
# ---------------------------------------------------------------------------


def test_write_data_yaml_correct_content(tmp_path):
    import yaml

    out_dir = tmp_path / "yolo_damage"
    COCOtoYOLOConverter.write_data_yaml(
        out_dir, nc=1, names=["damage"],
        train_path="train/images", val_path="valid/images", test_path="test/images",
    )
    yaml_file = out_dir / "data.yaml"
    assert yaml_file.exists()
    data = yaml.safe_load(yaml_file.read_text())
    assert data["nc"] == 1
    assert data["names"] == ["damage"]
    assert data["train"] == "train/images"
    assert data["val"] == "valid/images"
    assert data["test"] == "test/images"


# ---------------------------------------------------------------------------
# Test 15 — run_conversion_assertions: passes on clean output
# ---------------------------------------------------------------------------


def test_run_conversion_assertions_passes_clean(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "img_001.txt").write_text("0 0.500000 0.500000 0.156250 0.166667\n")
    (labels_dir / "img_002.txt").write_text("")  # empty is valid

    # Should not raise
    COCOtoYOLOConverter.run_conversion_assertions(tmp_path)


# ---------------------------------------------------------------------------
# Test 16 — run_conversion_assertions: raises on malformed label
# ---------------------------------------------------------------------------


def test_run_conversion_assertions_raises_on_malformed(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    # Missing the 5th token
    (labels_dir / "bad.txt").write_text("0 0.5 0.5 0.1\n")

    with pytest.raises(AssertionError, match="5 tokens"):
        COCOtoYOLOConverter.run_conversion_assertions(tmp_path)


# ---------------------------------------------------------------------------
# Test 17 — Full pipeline: synthetic COCO → convert_split → assertions
# ---------------------------------------------------------------------------


def test_full_pipeline_synthetic(tmp_path):
    """End-to-end test: build synthetic COCO, convert, assert, check data.yaml."""
    import yaml

    coco = _make_coco(n_images=3, img_w=640, img_h=480, with_annotations=True, category_ids=[1, 2])
    # Add a second annotation for image 1 with category 2
    coco["annotations"].append(
        {"id": 10, "image_id": 1, "category_id": 2, "bbox": [200, 100, 80, 60]}
    )

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for img in coco["images"]:
        (img_dir / img["file_name"]).write_bytes(b"FAKE")

    out_dir = tmp_path / "yolo_multiclass"
    category_map = {1: 0, 2: 1}
    class_names = ["class_1", "class_2"]

    COCOtoYOLOConverter.convert_split(coco, img_dir, out_dir, category_map, class_names)
    COCOtoYOLOConverter.write_data_yaml(
        out_dir, nc=2, names=class_names,
        train_path="train/images", val_path="valid/images", test_path="test/images",
    )
    # Assertions must pass
    COCOtoYOLOConverter.run_conversion_assertions(out_dir)

    # data.yaml check
    data = yaml.safe_load((out_dir / "data.yaml").read_text())
    assert data["nc"] == 2
    assert data["names"] == class_names

    # Every image has a label file
    label_files = {f.stem for f in (out_dir / "labels").glob("*.txt")}
    image_stems = {Path(img["file_name"]).stem for img in coco["images"]}
    assert image_stems == label_files
