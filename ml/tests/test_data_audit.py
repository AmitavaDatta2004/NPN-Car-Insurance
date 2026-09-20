"""Unit tests for Phase 1 data audit, quality checks, and manifest splitting."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from claimvision_ml.data.audit import (
    audit_image_files,
    compute_file_hashes,
    compute_perceptual_hashes,
    detect_shortcut_risks,
    find_exact_duplicate_groups,
    load_and_validate_csv,
)
from claimvision_ml.data.manifest import (
    create_group_aware_split,
    save_manifests,
    validate_split_leakage,
)
from claimvision_ml.quality.image_checks import (
    compute_image_metrics,
    read_image_safely,
)


@pytest.fixture
def sample_images_dir(tmp_path: Path) -> tuple[Path, list[Path]]:
    """Create a temporary directory with synthetic test images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Image 1: sharp black/white checkerboard
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img1[:50, :50] = 255
    img1[50:, 50:] = 255
    path1 = img_dir / "sharp.png"
    cv2.imwrite(str(path1), img1)

    # Image 2: blurry uniform image
    img2 = np.full((100, 100, 3), 128, dtype=np.uint8)
    path2 = img_dir / "blurry.png"
    cv2.imwrite(str(path2), img2)

    # Image 3: exact duplicate of image 1
    path3 = img_dir / "sharp_duplicate.png"
    cv2.imwrite(str(path3), img1)

    # Image 4: slightly modified (near duplicate of sharp)
    img4 = img1.copy()
    img4[0, 0] = 200
    path4 = img_dir / "sharp_near_dup.png"
    cv2.imwrite(str(path4), img4)

    return img_dir, [path1, path2, path3, path4]


def test_read_image_safely_valid(sample_images_dir: tuple[Path, list[Path]]) -> None:
    _, images = sample_images_dir
    img, meta = read_image_safely(images[0])
    assert img is not None
    assert meta["is_valid"] is True
    assert meta["width"] == 100
    assert meta["height"] == 100
    assert meta["channels"] == 3
    assert meta["error"] is None


def test_read_image_safely_invalid(tmp_path: Path) -> None:
    corrupt_file = tmp_path / "corrupt.jpg"
    corrupt_file.write_bytes(b"not_an_image_data")

    img, meta = read_image_safely(corrupt_file)
    assert img is None
    assert meta["is_valid"] is False
    assert meta["error"] == "corrupt_or_undecodable"


def test_image_quality_metrics(sample_images_dir: tuple[Path, list[Path]]) -> None:
    _, images = sample_images_dir
    sharp_path, blurry_path, _, _ = images

    sharp_metrics = compute_image_metrics(sharp_path)
    blurry_metrics = compute_image_metrics(blurry_path)

    assert sharp_metrics["blur_score"] > blurry_metrics["blur_score"]
    assert (
        blurry_metrics["blur_score"] == 0.0
    )  # constant image has 0 laplacian variance
    assert round(blurry_metrics["brightness"]) == 128
    assert round(blurry_metrics["contrast"]) == 0


def test_file_hashes_and_duplicates(sample_images_dir: tuple[Path, list[Path]]) -> None:
    _, images = sample_images_dir
    path1, path2, path3, _ = images

    hashes = compute_file_hashes([path1, path2, path3])
    assert hashes[str(path1)] == hashes[str(path3)]
    assert hashes[str(path1)] != hashes[str(path2)]

    dup_groups = find_exact_duplicate_groups(hashes)
    assert len(dup_groups) == 1
    assert set(dup_groups[0]) == {str(path1), str(path3)}


def test_perceptual_hashes(sample_images_dir: tuple[Path, list[Path]]) -> None:
    _, images = sample_images_dir
    path1, _, path3, path4 = images

    hash_dict, clusters = compute_perceptual_hashes([path1, path3, path4], threshold=5)
    assert len(hash_dict) == 3
    assert len(clusters) >= 1
    assert str(path1) in clusters[0] and str(path3) in clusters[0]


def test_audit_image_files(sample_images_dir: tuple[Path, list[Path]]) -> None:
    img_dir, images = sample_images_dir

    df = pd.DataFrame(
        {
            "filename": ["sharp.png", "blurry.png", "missing.png"],
            "label": ["genuine", "suspicious", "genuine"],
            "claim_id": ["C1", "C2", "C3"],
        }
    )

    audit_result = audit_image_files(df, img_dir, filename_col="filename")
    assert audit_result["matched_count"] == 2
    assert audit_result["missing_count"] == 1
    assert "missing.png" in audit_result["missing_files"]
    assert (
        audit_result["orphan_count"] >= 2
    )  # sharp_duplicate and sharp_near_dup are not in df


def test_detect_shortcut_risks() -> None:
    df = pd.DataFrame(
        {
            "label": ["genuine", "genuine", "suspicious", "suspicious"],
            "width": [1920, 1920, 800, 800],
            "height": [1080, 1080, 600, 600],
            "aspect_ratio": [1.77, 1.77, 1.33, 1.33],
            "blur_score": [120.0, 110.0, 30.0, 25.0],
            "brightness": [125.0, 130.0, 128.0, 127.0],
            "contrast": [55.0, 50.0, 52.0, 51.0],
        }
    )

    risks = detect_shortcut_risks(df, label_col="label")
    assert "width" in risks["disparities"]
    assert risks["disparities"]["width"]["shortcut_risk"] == "HIGH"
    assert risks["disparities"]["brightness"]["shortcut_risk"] == "LOW"


def test_group_aware_split_and_zero_leakage() -> None:
    # 20 samples belonging to 5 claims
    data = []
    for i in range(20):
        claim_id = f"CLAIM_{i // 4}"
        label = "genuine" if (i // 4) % 2 == 0 else "suspicious"
        data.append(
            {
                "path": f"/path/to/img_{i}.png",
                "claim_id": claim_id,
                "label": label,
                "sha256": f"hash_{i}",
            }
        )
    df = pd.DataFrame(data)

    train_df, val_df, test_df = create_group_aware_split(
        df,
        group_col="claim_id",
        ratios=(0.60, 0.20, 0.20),
        seed=42,
    )

    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    # Validate leakage
    report = validate_split_leakage(
        train_df, val_df, test_df, group_col="claim_id", hash_col="sha256"
    )
    assert report["zero_leakage_verified"] is True

    # Assert that no claim crosses split
    train_claims = set(train_df["claim_id"])
    val_claims = set(val_df["claim_id"])
    test_claims = set(test_df["claim_id"])
    assert not (train_claims & val_claims)
    assert not (train_claims & test_claims)
    assert not (val_claims & test_claims)


def test_leakage_detector_raises_on_contamination() -> None:
    df1 = pd.DataFrame(
        {"claim_id": ["C1", "C2"], "sha256": ["h1", "h2"], "path": ["p1", "p2"]}
    )
    df2 = pd.DataFrame(
        {"claim_id": ["C2", "C3"], "sha256": ["h3", "h4"], "path": ["p3", "p4"]}
    )  # C2 is in both!
    df3 = pd.DataFrame({"claim_id": ["C4"], "sha256": ["h5"], "path": ["p5"]})

    with pytest.raises(AssertionError, match="Split leakage detected"):
        validate_split_leakage(df1, df2, df3, group_col="claim_id")


def test_load_and_validate_csv(tmp_path: Path) -> None:
    csv_file = tmp_path / "test.csv"
    pd.DataFrame({"filename": ["a.jpg"], "label": ["genuine"]}).to_csv(
        csv_file, index=False
    )

    loaded = load_and_validate_csv(csv_file, expected_cols=["filename", "label"])
    assert len(loaded) == 1
    assert "filename" in loaded.columns

    with pytest.raises(ValueError, match="missing expected columns"):
        load_and_validate_csv(csv_file, expected_cols=["nonexistent_column"])


def test_save_manifests(tmp_path: Path) -> None:
    train_df = pd.DataFrame({"filename": ["1.jpg"], "label": ["genuine"]})
    val_df = pd.DataFrame({"filename": ["2.jpg"], "label": ["suspicious"]})
    test_df = pd.DataFrame({"filename": ["3.jpg"], "label": ["genuine"]})

    out_dir = tmp_path / "manifests"
    paths = save_manifests(train_df, val_df, test_df, output_dir=out_dir, prefix="test")

    assert Path(paths["train"]).exists()
    assert Path(paths["val"]).exists()
    assert Path(paths["test"]).exists()
    assert Path(paths["summary"]).exists()
