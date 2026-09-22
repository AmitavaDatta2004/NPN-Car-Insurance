"""test_location_dataset.py — Unit tests for the location dataset module.

Task ID  : LOC-DATA-001
Phase    : 10b

Tests cover:
- LOCATION_CLASSES constant correctness
- derive_location_labels: dominant-part selection by count
- derive_location_labels: area tiebreak
- derive_location_labels: missing file handling
- derive_location_labels: empty annotations
- derive_location_labels: unknown category is skipped
- get_location_transforms: output shapes for train / val / test
- LocationDataset: label_counts covers all 5 classes
- LocationDataset: class_weights shape
- LocationDataset: raises RuntimeError on empty directory
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_location_classes_length():
    from claimvision_ml.location.dataset import LOCATION_CLASSES
    assert len(LOCATION_CLASSES) == 5


def test_location_classes_values():
    from claimvision_ml.location.dataset import LOCATION_CLASSES
    assert LOCATION_CLASSES == ["headlamp", "front_bumper", "hood", "door", "rear_bumper"]


def test_location_class_to_id_consistency():
    from claimvision_ml.location.dataset import LOCATION_CLASSES, LOCATION_CLASS_TO_ID, LOCATION_ID_TO_CLASS
    for idx, name in enumerate(LOCATION_CLASSES):
        assert LOCATION_CLASS_TO_ID[name] == idx
        assert LOCATION_ID_TO_CLASS[idx] == name


# ---------------------------------------------------------------------------
# Helpers to create synthetic COCO JSON
# ---------------------------------------------------------------------------

def _make_coco_json(image_list, annotation_list, category_list):
    return {
        "images": image_list,
        "annotations": annotation_list,
        "categories": category_list,
    }


def _write_coco_json(data: dict, path: Path) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# derive_location_labels
# ---------------------------------------------------------------------------

def test_derive_dominant_by_count(tmp_path):
    """Image with 3 headlamp boxes and 1 door box → label = headlamp."""
    from claimvision_ml.location.dataset import derive_location_labels

    data = _make_coco_json(
        image_list=[{"id": 1, "file_name": "img1.jpg"}],
        category_list=[
            {"id": 10, "name": "headlamp"},
            {"id": 20, "name": "door"},
        ],
        annotation_list=[
            {"id": 1, "image_id": 1, "category_id": 10, "bbox": [0, 0, 100, 100]},
            {"id": 2, "image_id": 1, "category_id": 10, "bbox": [10, 10, 50, 50]},
            {"id": 3, "image_id": 1, "category_id": 10, "bbox": [20, 20, 30, 30]},
            {"id": 4, "image_id": 1, "category_id": 20, "bbox": [0, 0, 200, 200]},
        ],
    )
    json_path = _write_coco_json(data, tmp_path / "anno.json")
    result = derive_location_labels(json_path)

    assert "img1.jpg" in result
    label_id, label_name = result["img1.jpg"]
    assert label_name == "headlamp"
    assert label_id == 0


def test_derive_area_tiebreak(tmp_path):
    """Image with 1 headlamp box (small) and 1 door box (large) → label = door (area wins)."""
    from claimvision_ml.location.dataset import derive_location_labels

    data = _make_coco_json(
        image_list=[{"id": 1, "file_name": "img2.jpg"}],
        category_list=[
            {"id": 1, "name": "headlamp"},
            {"id": 2, "name": "door"},
        ],
        annotation_list=[
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},   # area 100
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [0, 0, 200, 200]}, # area 40000
        ],
    )
    json_path = _write_coco_json(data, tmp_path / "anno.json")
    result = derive_location_labels(json_path)
    label_id, label_name = result["img2.jpg"]
    assert label_name == "door"


def test_derive_skips_image_with_no_recognised_annotations(tmp_path):
    """Image with only 'damage' category (not in LOCATION_CLASSES) should be skipped."""
    from claimvision_ml.location.dataset import derive_location_labels

    data = _make_coco_json(
        image_list=[{"id": 1, "file_name": "img3.jpg"}],
        category_list=[{"id": 99, "name": "damage"}],
        annotation_list=[
            {"id": 1, "image_id": 1, "category_id": 99, "bbox": [0, 0, 100, 100]},
        ],
    )
    json_path = _write_coco_json(data, tmp_path / "anno.json")
    result = derive_location_labels(json_path)
    assert "img3.jpg" not in result


def test_derive_empty_annotations(tmp_path):
    """Image with zero annotations should be absent from result."""
    from claimvision_ml.location.dataset import derive_location_labels

    data = _make_coco_json(
        image_list=[{"id": 1, "file_name": "img4.jpg"}],
        category_list=[{"id": 1, "name": "headlamp"}],
        annotation_list=[],
    )
    json_path = _write_coco_json(data, tmp_path / "anno.json")
    result = derive_location_labels(json_path)
    assert len(result) == 0


def test_derive_normalises_space_name(tmp_path):
    """Category name 'front bumper' (with space) should be mapped to 'front_bumper'."""
    from claimvision_ml.location.dataset import derive_location_labels

    data = _make_coco_json(
        image_list=[{"id": 1, "file_name": "img5.jpg"}],
        category_list=[{"id": 5, "name": "front bumper"}],
        annotation_list=[
            {"id": 1, "image_id": 1, "category_id": 5, "bbox": [0, 0, 100, 100]},
        ],
    )
    json_path = _write_coco_json(data, tmp_path / "anno.json")
    result = derive_location_labels(json_path)
    _, label_name = result["img5.jpg"]
    assert label_name == "front_bumper"


def test_derive_missing_json_raises():
    from claimvision_ml.location.dataset import derive_location_labels
    with pytest.raises(FileNotFoundError):
        derive_location_labels("/nonexistent/path/anno.json")


def test_derive_multiple_images(tmp_path):
    """Multiple images with different dominant parts all get correct labels."""
    from claimvision_ml.location.dataset import derive_location_labels

    data = _make_coco_json(
        image_list=[
            {"id": 1, "file_name": "a.jpg"},
            {"id": 2, "file_name": "b.jpg"},
        ],
        category_list=[
            {"id": 1, "name": "headlamp"},
            {"id": 2, "name": "hood"},
        ],
        annotation_list=[
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 50, 50]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [0, 0, 50, 50]},
            {"id": 3, "image_id": 2, "category_id": 2, "bbox": [0, 0, 100, 100]},
        ],
    )
    json_path = _write_coco_json(data, tmp_path / "anno.json")
    result = derive_location_labels(json_path)
    assert result["a.jpg"][1] == "headlamp"
    assert result["b.jpg"][1] == "hood"


# ---------------------------------------------------------------------------
# get_location_transforms
# ---------------------------------------------------------------------------

def test_transforms_output_shape_train():
    from claimvision_ml.location.dataset import get_location_transforms
    tf = get_location_transforms("train", image_size=224)
    img = Image.new("RGB", (300, 200))
    tensor = tf(img)
    assert tensor.shape == (3, 224, 224)


def test_transforms_output_shape_val():
    from claimvision_ml.location.dataset import get_location_transforms
    tf = get_location_transforms("val", image_size=224)
    img = Image.new("RGB", (300, 200))
    tensor = tf(img)
    assert tensor.shape == (3, 224, 224)


def test_transforms_output_shape_test():
    from claimvision_ml.location.dataset import get_location_transforms
    tf = get_location_transforms("test", image_size=224)
    img = Image.new("RGB", (300, 200))
    tensor = tf(img)
    assert tensor.shape == (3, 224, 224)


def test_transforms_invalid_split():
    from claimvision_ml.location.dataset import get_location_transforms
    with pytest.raises(ValueError):
        get_location_transforms("predict")


# ---------------------------------------------------------------------------
# LocationDataset
# ---------------------------------------------------------------------------

def _create_fake_dataset(tmp_path: Path, n_images: int = 5) -> tuple[Path, Path]:
    """Create fake images + COCO JSON for LocationDataset testing."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    categories = [
        {"id": 1, "name": "headlamp"},
        {"id": 2, "name": "front_bumper"},
        {"id": 3, "name": "hood"},
        {"id": 4, "name": "door"},
        {"id": 5, "name": "rear_bumper"},
    ]
    images = []
    annotations = []
    ann_id = 1
    for i in range(n_images):
        img = Image.new("RGB", (224, 224), color=(i * 30 % 255, 100, 200))
        fname = f"img_{i:03d}.jpg"
        img.save(img_dir / fname)
        images.append({"id": i + 1, "file_name": fname})
        cat_id = (i % 5) + 1
        annotations.append({
            "id": ann_id,
            "image_id": i + 1,
            "category_id": cat_id,
            "bbox": [0, 0, 50, 50],
        })
        ann_id += 1

    coco_data = {"images": images, "annotations": annotations, "categories": categories}
    json_path = tmp_path / "anno.json"
    json_path.write_text(json.dumps(coco_data), encoding="utf-8")
    return img_dir, json_path


def test_location_dataset_length(tmp_path):
    from claimvision_ml.location.dataset import LocationDataset
    img_dir, json_path = _create_fake_dataset(tmp_path, n_images=5)
    ds = LocationDataset(img_dir, json_path, split="train")
    assert len(ds) == 5


def test_location_dataset_item_shapes(tmp_path):
    from claimvision_ml.location.dataset import LocationDataset
    img_dir, json_path = _create_fake_dataset(tmp_path, n_images=5)
    ds = LocationDataset(img_dir, json_path, split="val")
    tensor, label = ds[0]
    assert tensor.shape == (3, 224, 224)
    assert label.dtype == torch.long
    assert 0 <= int(label.item()) <= 4


def test_location_dataset_class_weights_shape(tmp_path):
    from claimvision_ml.location.dataset import LocationDataset
    img_dir, json_path = _create_fake_dataset(tmp_path, n_images=5)
    ds = LocationDataset(img_dir, json_path, split="train")
    weights = ds.class_weights()
    assert weights.shape == (5,)
    assert (weights > 0).all()


def test_location_dataset_empty_dir_raises(tmp_path):
    """Empty image directory with no matching files should raise RuntimeError."""
    from claimvision_ml.location.dataset import LocationDataset
    img_dir = tmp_path / "empty"
    img_dir.mkdir()

    categories = [{"id": 1, "name": "headlamp"}]
    images = [{"id": 1, "file_name": "nonexistent.jpg"}]
    annotations = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 50, 50]}]
    coco_data = {"images": images, "annotations": annotations, "categories": categories}
    json_path = tmp_path / "anno.json"
    json_path.write_text(json.dumps(coco_data), encoding="utf-8")

    with pytest.raises(RuntimeError):
        LocationDataset(img_dir, json_path, split="train")


def test_location_dataset_from_explicit_samples(tmp_path):
    from claimvision_ml.location.dataset import LocationDataset
    img_path = tmp_path / "sample.jpg"
    Image.new("RGB", (100, 100)).save(img_path)
    samples = [(img_path, 2)]
    ds = LocationDataset(samples=samples, split="train")
    assert len(ds) == 1
    tensor, label = ds[0]
    assert tensor.shape == (3, 224, 224)
    assert label.item() == 2


def test_location_dataset_class_weights_normalized(tmp_path):
    from claimvision_ml.location.dataset import LocationDataset
    img_dir, json_path = _create_fake_dataset(tmp_path, n_images=10)
    ds = LocationDataset(img_dir, json_path, split="train")
    weights = ds.class_weights()
    assert float(weights.mean()) == pytest.approx(1.0, rel=1e-3)


def test_load_location_splits_sparse_val_pooling(tmp_path):
    """When val split has < 5 images, load_location_splits pools all annotated images and stratifies."""
    from claimvision_ml.location.dataset import load_location_splits

    # Create fake raw_coco_dir with train (15 images) and val (1 image)
    raw_dir = tmp_path / "raw_coco"
    train_dir = raw_dir / "train"
    val_dir = raw_dir / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    # 15 train images (3 per class)
    train_images = []
    train_annotations = []
    categories = [
        {"id": 1, "name": "headlamp"},
        {"id": 2, "name": "front_bumper"},
        {"id": 3, "name": "hood"},
        {"id": 4, "name": "door"},
        {"id": 5, "name": "rear_bumper"},
    ]
    ann_id = 1
    for i in range(15):
        fname = f"train_{i}.jpg"
        Image.new("RGB", (100, 100)).save(train_dir / fname)
        train_images.append({"id": i + 1, "file_name": fname})
        cat_id = (i % 5) + 1
        train_annotations.append({"id": ann_id, "image_id": i + 1, "category_id": cat_id, "bbox": [0, 0, 40, 40]})
        ann_id += 1

    (train_dir / "COCO_mul_train_annos.json").write_text(
        json.dumps({"images": train_images, "annotations": train_annotations, "categories": categories}),
        encoding="utf-8",
    )

    # 1 val image
    Image.new("RGB", (100, 100)).save(val_dir / "val_0.jpg")
    val_images = [{"id": 1, "file_name": "val_0.jpg"}]
    val_annotations = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 40, 40]}]
    (val_dir / "COCO_mul_val_annos.json").write_text(
        json.dumps({"images": val_images, "annotations": val_annotations, "categories": categories}),
        encoding="utf-8",
    )

    train_ds, val_ds = load_location_splits(raw_dir, val_ratio=0.25, seed=42)
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(train_ds) + len(val_ds) == 16
    # Check that val has at least 1 image per class because of stratification
    assert len(val_ds) == 5
