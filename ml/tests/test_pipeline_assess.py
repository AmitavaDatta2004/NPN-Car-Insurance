"""Integration and unit tests for unified claim assessment pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from claimvision_ml.pipeline import (
    AssessmentResult,
    assess_claim,
)


@pytest.fixture
def sample_valid_image(tmp_path: Path) -> Path:
    """Create a synthetic high-variance image that passes quality checks."""
    img_path = tmp_path / "valid_vehicle.jpg"
    # Create textured image with strong edges to guarantee blur variance > 50
    arr = np.zeros((300, 300, 3), dtype=np.uint8)
    for i in range(300):
        for j in range(300):
            arr[i, j] = [(i * 7) % 256, (j * 11) % 256, ((i + j) * 5) % 256]
    img = Image.fromarray(arr)
    img.save(img_path, format="JPEG", quality=95)
    return img_path


@pytest.fixture
def sample_tiny_image(tmp_path: Path) -> Path:
    """Create a tiny image below minimum resolution (50x50)."""
    img_path = tmp_path / "tiny_image.jpg"
    img = Image.new("RGB", (50, 50), color=(100, 100, 100))
    img.save(img_path, format="JPEG")
    return img_path


def test_assess_claim_file_not_found(tmp_path: Path):
    """Verify non-existent file returns MORE_EVIDENCE_REQUIRED."""
    non_existent = tmp_path / "ghost.jpg"
    res = assess_claim(non_existent)

    assert isinstance(res, AssessmentResult)
    assert res.route == "MORE_EVIDENCE_REQUIRED"
    assert "image_file_not_found" in res.reason_codes


def test_assess_claim_quality_failure(sample_tiny_image: Path):
    """Verify sub-resolution image fails quality and returns MORE_EVIDENCE_REQUIRED."""
    res = assess_claim(sample_tiny_image)

    assert res.route == "MORE_EVIDENCE_REQUIRED"
    assert res.quality is not None
    assert res.quality.acceptable is False
    # Downstream models should not have run
    assert res.fraud is None
    assert res.severity is None
    assert res.cost is None


def test_assess_claim_valid_flow(sample_valid_image: Path):
    """Verify end-to-end flow with a valid image produces complete AssessmentResult."""
    res = assess_claim(sample_valid_image, claim_id="test-claim-123", vehicle_segment="sedan")

    assert res.claim_id == "test-claim-123"
    assert res.quality is not None
    assert res.quality.acceptable is True
    assert res.fraud is not None
    assert res.severity is not None
    assert res.location is not None
    assert res.cost is not None
    assert len(res.detections) > 0
    assert res.cost.vehicle_segment == "sedan"
    assert res.route in ("FAST_TRACK_ELIGIBLE", "MANUAL_DAMAGE_REVIEW", "FRAUD_REVIEW")
    assert res.inference_ms > 0
    assert "fraud" in res.model_versions
    assert "severity" in res.model_versions
    assert "location" in res.model_versions


def test_assess_claim_high_fraud_early_exit(sample_valid_image: Path):
    """Verify high fraud probability skips damage, location, and cost stages."""
    # Set threshold very low so default fraud prob (0.12) exceeds it
    custom_cfg = {
        "thresholds": {
            "fraud": {"low_threshold": 0.05, "high_threshold": 0.10},
            "severity": {"confidence_min": 0.40},
            "location": {"confidence_min": 0.40},
            "routing": {"fast_track_max_cost": 50000.0},
        }
    }
    res = assess_claim(sample_valid_image, config=custom_cfg)

    assert res.route == "FRAUD_REVIEW"
    assert "high_fraud_risk" in res.reason_codes
    assert res.fraud is not None
    assert res.severity is None
    assert res.location is None
    assert res.cost is None
    assert len(res.detections) == 0


def test_assessment_result_serialization(sample_valid_image: Path):
    """Verify to_dict() returns valid dictionary matching API contract."""
    res = assess_claim(sample_valid_image)
    d = res.to_dict()

    assert isinstance(d, dict)
    assert d["claim_id"] == res.claim_id
    assert d["route"] == res.route
    assert isinstance(d["reason_codes"], list)
    assert isinstance(d["quality"], dict)
    assert isinstance(d["fraud"], dict)
    assert isinstance(d["severity"], dict)
    assert isinstance(d["location"], dict)
    assert isinstance(d["detections"], list)
    assert isinstance(d["cost"], dict)
    assert isinstance(d["model_versions"], dict)
    assert isinstance(d["inference_ms"], float)
