"""Unit tests for severity dataset audit, deduplication, and manifest generation."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from claimvision_ml.data.severity_audit import (
    audit_severity_images,
    compute_file_sha256,
    create_severity_split,
    discover_severity_images,
    find_exact_duplicate_groups,
    normalize_class_name,
    save_severity_manifests,
    validate_severity_split_leakage,
)


@pytest.fixture
def mock_severity_dataset(tmp_path: Path) -> Path:
    """Create a minimal mock dataset mimicking the prajwalbhamere / data3a layout."""
    dataset_dir = tmp_path / "car_damage_severity" / "data3a"

    counter = 0
    for split in ["training", "validation"]:
        for cls_name in ["01-minor", "02-moderate", "03-severe"]:
            folder = dataset_dir / split / cls_name
            folder.mkdir(parents=True, exist_ok=True)

            # Create 4 synthetic images per category with unique content
            for i in range(4):
                counter += 1
                img = np.full((100, 100, 3), fill_value=(counter * 5) % 250, dtype=np.uint8)
                cv2.rectangle(
                    img, (10 + i, 10 + i), (80, 80), ((counter * 7) % 255, 150, 100), -1
                )
                cv2.putText(
                    img,
                    f"{cls_name}_{i}_{split}",
                    (5, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1,
                )
                img_path = folder / f"img_{i}.jpg"
                cv2.imwrite(str(img_path), img)

    return tmp_path / "car_damage_severity"


def test_normalize_class_name():
    assert normalize_class_name("01-minor") == "minor"
    assert normalize_class_name("minor") == "minor"
    assert normalize_class_name("02-moderate") == "moderate"
    assert normalize_class_name("moderate") == "moderate"
    assert normalize_class_name("03-severe") == "severe"
    assert normalize_class_name("severe") == "severe"

    with pytest.raises(ValueError, match="Unrecognized severity"):
        normalize_class_name("04-unknown")


def test_discover_severity_images(mock_severity_dataset: Path):
    df = discover_severity_images(mock_severity_dataset)
    assert len(df) == 24  # 2 splits * 3 classes * 4 images
    assert set(df["label"].unique()) == {"minor", "moderate", "severe"}
    assert set(df["label_id"].unique()) == {0, 1, 2}
    assert "image_path" in df.columns
    assert "filename" in df.columns
    assert "original_split" in df.columns


def test_compute_file_sha256(tmp_path: Path):
    f = tmp_path / "test.txt"
    f.write_text("claimvision-ai-severity")
    h1 = compute_file_sha256(f)
    h2 = compute_file_sha256(f)
    assert h1 == h2
    assert len(h1) == 64


def test_audit_severity_images(tmp_path: Path):
    # 1 valid image and 1 corrupt image
    valid_file = tmp_path / "valid.jpg"
    img = np.ones((64, 64, 3), dtype=np.uint8) * 128
    cv2.imwrite(str(valid_file), img)

    corrupt_file = tmp_path / "corrupt.jpg"
    corrupt_file.write_bytes(b"not an image file content")

    df = pd.DataFrame(
        [
            {
                "image_path": str(valid_file),
                "filename": "valid.jpg",
                "label": "minor",
                "label_id": 0,
            },
            {
                "image_path": str(corrupt_file),
                "filename": "corrupt.jpg",
                "label": "severe",
                "label_id": 2,
            },
        ]
    )

    audit_df = audit_severity_images(df)
    assert len(audit_df) == 2

    valid_row = audit_df[audit_df["filename"] == "valid.jpg"].iloc[0]
    assert not valid_row["is_corrupt"]
    assert valid_row["width"] == 64
    assert valid_row["height"] == 64
    assert len(valid_row["sha256"]) == 64
    assert valid_row["phash"] != ""

    corrupt_row = audit_df[audit_df["filename"] == "corrupt.jpg"].iloc[0]
    assert bool(corrupt_row["is_corrupt"]) is True
    assert corrupt_row["width"] == 0



def test_find_exact_duplicate_groups(tmp_path: Path):
    img = np.full((50, 50, 3), 100, dtype=np.uint8)
    p1 = tmp_path / "p1.jpg"
    p2 = tmp_path / "p2.jpg"
    p3 = tmp_path / "p3.jpg"

    cv2.imwrite(str(p1), img)
    cv2.imwrite(str(p2), img)  # exact copy
    cv2.imwrite(str(p3), img + 20)  # different

    df = pd.DataFrame(
        [
            {
                "image_path": str(p1),
                "label": "minor",
                "is_corrupt": False,
                "sha256": compute_file_sha256(p1),
            },
            {
                "image_path": str(p2),
                "label": "minor",
                "is_corrupt": False,
                "sha256": compute_file_sha256(p2),
            },
            {
                "image_path": str(p3),
                "label": "moderate",
                "is_corrupt": False,
                "sha256": compute_file_sha256(p3),
            },
        ]
    )

    dupes = find_exact_duplicate_groups(df)
    assert len(dupes) == 1
    paths = list(dupes.values())[0]
    assert set(paths) == {str(p1), str(p2)}


def test_create_severity_split_and_validation(mock_severity_dataset: Path):
    df = discover_severity_images(mock_severity_dataset)
    audit_df = audit_severity_images(df)

    train_df, val_df, test_df = create_severity_split(
        audit_df, ratios=(0.70, 0.15, 0.15), seed=42
    )

    assert len(train_df) + len(val_df) + len(test_df) == len(audit_df)
    assert set(train_df["label"].unique()) == {"minor", "moderate", "severe"}
    assert set(val_df["label"].unique()) == {"minor", "moderate", "severe"}
    assert set(test_df["label"].unique()) == {"minor", "moderate", "severe"}

    # Validation should pass with zero leakage
    leakage_res = validate_severity_split_leakage(train_df, val_df, test_df)
    assert leakage_res["zero_leakage"] is True
    assert leakage_res["status"] == "PASSED"


def test_validate_severity_split_leakage_raises_on_contamination():
    sample_data = {
        "image_path": ["a.jpg", "b.jpg"],
        "sha256": ["hash_a", "hash_b"],
        "label": ["minor", "moderate"],
    }
    train_df = pd.DataFrame(sample_data)
    # val_df contains hash_a (leakage)
    val_df = pd.DataFrame(
        {
            "image_path": ["c.jpg"],
            "sha256": ["hash_a"],
            "label": ["minor"],
        }
    )
    test_df = pd.DataFrame(
        {
            "image_path": ["d.jpg"],
            "sha256": ["hash_d"],
            "label": ["severe"],
        }
    )

    with pytest.raises(AssertionError, match="hash overlap"):
        validate_severity_split_leakage(train_df, val_df, test_df)


def test_save_severity_manifests(tmp_path: Path):
    train_df = pd.DataFrame(
        [
            {
                "image_path": "data/1.jpg",
                "filename": "1.jpg",
                "label": "minor",
                "label_id": 0,
                "sha256": "h1",
                "split": "train",
            }
        ]
    )
    val_df = pd.DataFrame(
        [
            {
                "image_path": "data/2.jpg",
                "filename": "2.jpg",
                "label": "moderate",
                "label_id": 1,
                "sha256": "h2",
                "split": "val",
            }
        ]
    )
    test_df = pd.DataFrame(
        [
            {
                "image_path": "data/3.jpg",
                "filename": "3.jpg",
                "label": "severe",
                "label_id": 2,
                "sha256": "h3",
                "split": "test",
            }
        ]
    )

    manifest_dir = tmp_path / "manifests"
    saved = save_severity_manifests(
        train_df, val_df, test_df, output_dir=manifest_dir
    )

    assert Path(saved["train"]).exists()
    assert Path(saved["val"]).exists()
    assert Path(saved["test"]).exists()
    assert Path(saved["class_map"]).exists()
    assert Path(saved["summary"]).exists()

    with open(saved["class_map"]) as f:
        cmap = json.load(f)
    assert cmap == {"0": "minor", "1": "moderate", "2": "severe"}
