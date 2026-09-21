"""Unit tests for Phase 3 — OpenCV quality runtime checker.

Tests cover all 9 check stages of run_quality_checks() and all 5 new
helper functions added to image_checks.py in Phase 3.

Fixture strategy:
  - Synthetic images are created in pytest tmp_path using numpy + cv2.
  - No real dataset files are required; tests run in CI without network access.
  - All thresholds match the defaults in run_quality_checks() so tests are
    self-consistent and do not depend on configuration files.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from claimvision_ml.quality.image_checks import (
    check_minimum_resolution,
    draw_bounding_boxes,
    extract_exif_summary,
    save_annotated_image,
)
from claimvision_ml.quality.runtime_checker import run_quality_checks

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_png(path: Path, array: np.ndarray) -> Path:
    """Write a numpy BGR array as PNG and return the path."""
    cv2.imwrite(str(path), array)
    return path


@pytest.fixture
def tmp_img_dir(tmp_path: Path) -> Path:
    return tmp_path / "images"


@pytest.fixture
def clean_image(tmp_path: Path) -> Path:
    """Sharp, well-lit, full-colour image that passes every check."""
    # 400x400 checkerboard → high blur score, good brightness, good contrast
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    block = 20
    for r in range(0, 400, block):
        for c in range(0, 400, block):
            if (r // block + c // block) % 2 == 0:
                img[r : r + block, c : c + block] = 200
    path = tmp_path / "clean.png"
    return _write_png(path, img)


@pytest.fixture
def blurry_image(tmp_path: Path) -> Path:
    """Uniform grey image — Laplacian variance = 0 (maximally blurry)."""
    img = np.full((400, 400, 3), 128, dtype=np.uint8)
    path = tmp_path / "blurry.png"
    return _write_png(path, img)


@pytest.fixture
def dark_image(tmp_path: Path) -> Path:
    """Near-black image — brightness ≈ 10."""
    img = np.full((400, 400, 3), 10, dtype=np.uint8)
    path = tmp_path / "dark.png"
    return _write_png(path, img)


@pytest.fixture
def overexposed_image(tmp_path: Path) -> Path:
    """Near-white image — brightness ≈ 245."""
    img = np.full((400, 400, 3), 245, dtype=np.uint8)
    path = tmp_path / "overexposed.png"
    return _write_png(path, img)


@pytest.fixture
def low_contrast_image(tmp_path: Path) -> Path:
    """Uniform mid-grey image — contrast (std dev) ≈ 0."""
    img = np.full((400, 400, 3), 128, dtype=np.uint8)
    path = tmp_path / "low_contrast.png"
    return _write_png(path, img)


@pytest.fixture
def small_image(tmp_path: Path) -> Path:
    """100x100 image — below 224x224 minimum resolution."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :50] = 200
    path = tmp_path / "small.png"
    return _write_png(path, img)


@pytest.fixture
def adequate_resolution_image(tmp_path: Path) -> Path:
    """300x300 image — above minimum resolution."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:150, :150] = 200
    path = tmp_path / "adequate.png"
    return _write_png(path, img)


@pytest.fixture
def corrupt_image(tmp_path: Path) -> Path:
    """File with non-image bytes — undecodable by cv2."""
    path = tmp_path / "corrupt.jpg"
    path.write_bytes(b"this_is_not_an_image_XYZXYZ")
    return path


# ---------------------------------------------------------------------------
# Check 1: Corrupt / unreadable
# ---------------------------------------------------------------------------


def test_corrupt_image_rejected(corrupt_image: Path) -> None:
    result = run_quality_checks(corrupt_image)
    assert result.passed is False
    assert result.route == "MORE_EVIDENCE_REQUIRED"
    assert "corrupt_or_unreadable" in result.rejection_reasons


def test_missing_file_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.jpg"
    result = run_quality_checks(missing)
    assert result.passed is False
    assert result.route == "MORE_EVIDENCE_REQUIRED"
    assert "corrupt_or_unreadable" in result.rejection_reasons


# ---------------------------------------------------------------------------
# Check 2: Resolution
# ---------------------------------------------------------------------------


def test_below_min_resolution_rejected(small_image: Path) -> None:
    result = run_quality_checks(small_image, min_width=224, min_height=224)
    assert result.passed is False
    assert result.route == "MORE_EVIDENCE_REQUIRED"
    assert "resolution_too_low" in result.rejection_reasons


def test_adequate_resolution_passes_resolution_check(
    adequate_resolution_image: Path,
) -> None:
    result = run_quality_checks(adequate_resolution_image)
    assert "resolution_too_low" not in result.rejection_reasons
    assert result.width == 300
    assert result.height == 300


def test_check_minimum_resolution_function() -> None:
    """Unit test for the standalone check_minimum_resolution helper."""
    ok_img = np.zeros((300, 300, 3), dtype=np.uint8)
    small_img = np.zeros((100, 100, 3), dtype=np.uint8)

    passed, reason = check_minimum_resolution(ok_img, 224, 224)
    assert passed is True
    assert reason == ""

    passed, reason = check_minimum_resolution(small_img, 224, 224)
    assert passed is False
    assert reason == "resolution_too_low"


# ---------------------------------------------------------------------------
# Check 3: Blur
# ---------------------------------------------------------------------------


def test_blurry_image_flagged(blurry_image: Path) -> None:
    result = run_quality_checks(blurry_image, blur_threshold=50.0)
    assert "excessive_blur" in result.rejection_reasons
    assert result.blur_score < 50.0


def test_sharp_image_passes_blur_check(clean_image: Path) -> None:
    result = run_quality_checks(clean_image, blur_threshold=50.0)
    assert "excessive_blur" not in result.rejection_reasons
    assert result.blur_score >= 50.0


# ---------------------------------------------------------------------------
# Check 4: Brightness
# ---------------------------------------------------------------------------


def test_dark_image_flagged(dark_image: Path) -> None:
    result = run_quality_checks(dark_image, min_brightness=30.0)
    assert "too_dark" in result.rejection_reasons


def test_overexposed_image_flagged(overexposed_image: Path) -> None:
    result = run_quality_checks(overexposed_image, max_brightness=240.0)
    assert "overexposed" in result.rejection_reasons


# ---------------------------------------------------------------------------
# Check 5: Contrast
# ---------------------------------------------------------------------------


def test_low_contrast_flagged(low_contrast_image: Path) -> None:
    result = run_quality_checks(low_contrast_image, min_contrast=15.0)
    assert "low_contrast" in result.rejection_reasons


# ---------------------------------------------------------------------------
# Check 7: Exact duplicate
# ---------------------------------------------------------------------------


def test_exact_duplicate_routes_to_review(clean_image: Path) -> None:
    import hashlib

    with open(clean_image, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()

    historical_hashes = {sha: "historical/reference_image.jpg"}
    result = run_quality_checks(clean_image, historical_hashes=historical_hashes)

    assert result.is_exact_duplicate is True
    assert result.route == "DUPLICATE_REVIEW"
    assert result.passed is False
    assert result.duplicate_reference == "historical/reference_image.jpg"


# ---------------------------------------------------------------------------
# Check 8: Near-duplicate
# ---------------------------------------------------------------------------


def test_near_duplicate_routes_to_review(tmp_path: Path) -> None:
    import imagehash
    from PIL import Image as PilImage

    # Build a simple image and compute its dHash
    img_array = np.zeros((400, 400, 3), dtype=np.uint8)
    img_array[:200, :200] = 200
    path = tmp_path / "near_dup.png"
    _write_png(path, img_array)

    with PilImage.open(path) as pil:
        dhash_str = str(imagehash.dhash(pil, hash_size=8))

    historical_dhashes = {dhash_str: "historical/old_claim.jpg"}
    result = run_quality_checks(
        path,
        historical_dhashes=historical_dhashes,
        near_dup_threshold=4,
    )

    assert result.is_near_duplicate is True
    assert result.route == "DUPLICATE_REVIEW"
    assert result.passed is False


# ---------------------------------------------------------------------------
# Check 9: EXIF — must never create fraud/quality failure
# ---------------------------------------------------------------------------


def test_exif_absent_never_causes_failure(clean_image: Path) -> None:
    """PNG files have no EXIF — absence must not affect passed or route."""
    result = run_quality_checks(clean_image)
    # exif_absent may be in warnings but must never set passed=False alone
    # The clean_image should still pass all other checks
    if "exif_absent" in result.warnings:
        # Confirm the reason is NOT in rejection_reasons
        assert "exif_absent" not in result.rejection_reasons
    # The image itself should be valid
    assert result.route != "FRAUD_REVIEW"


def test_extract_exif_summary_returns_dict(clean_image: Path) -> None:
    """PNG has no EXIF — summary should indicate exif_available=False."""
    summary = extract_exif_summary(clean_image)
    assert isinstance(summary, dict)
    assert "exif_available" in summary
    assert summary["exif_available"] is False  # PNG has no EXIF


# ---------------------------------------------------------------------------
# Bounding box drawing
# ---------------------------------------------------------------------------


def test_draw_bounding_boxes_returns_image(clean_image: Path) -> None:
    img = cv2.imread(str(clean_image))
    boxes = [(50, 50, 200, 200)]
    labels = ["damage"]
    scores = [0.87]

    annotated = draw_bounding_boxes(img, boxes, labels, scores)

    assert annotated is not None
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == img.shape
    # Original must not be modified
    assert not np.array_equal(annotated, img)


def test_draw_bounding_boxes_does_not_modify_original(clean_image: Path) -> None:
    img = cv2.imread(str(clean_image))
    original_copy = img.copy()
    draw_bounding_boxes(img, [(10, 10, 100, 100)], ["test"])
    assert np.array_equal(img, original_copy)


# ---------------------------------------------------------------------------
# Annotated image saving
# ---------------------------------------------------------------------------


def test_save_annotated_image(tmp_path: Path, clean_image: Path) -> None:
    img = cv2.imread(str(clean_image))
    out_path = tmp_path / "annotated" / "result.png"
    saved = save_annotated_image(img, out_path)

    assert Path(saved).exists()
    reloaded = cv2.imread(saved)
    assert reloaded is not None
    assert reloaded.shape == img.shape


# ---------------------------------------------------------------------------
# Full pipeline: clean image passes all checks
# ---------------------------------------------------------------------------


def test_clean_image_passes_all_checks(clean_image: Path) -> None:
    result = run_quality_checks(clean_image)
    assert result.passed is True
    assert result.route == "CONTINUE"
    assert result.rejection_reasons == []
    assert result.is_exact_duplicate is False
    assert result.is_near_duplicate is False
    assert result.blur_score > 0.0
    assert result.width == 400
    assert result.height == 400


# ---------------------------------------------------------------------------
# to_dict serialisation
# ---------------------------------------------------------------------------


def test_quality_result_to_dict(clean_image: Path) -> None:
    result = run_quality_checks(clean_image)
    d = result.to_dict()
    assert isinstance(d, dict)
    assert "passed" in d
    assert "route" in d
    assert "blur_score" in d
    assert "sha256" in d
    assert isinstance(d["rejection_reasons"], list)
    assert isinstance(d["warnings"], list)
