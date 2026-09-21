"""test_damage_detector.py — Unit tests for Generic Damage Detector (DET-YOLO-001).

Phase: 9
Task: DET-YOLO-001
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

from claimvision_ml.detection.damage import (
    DamageDetection,
    DamageDetector,
    detect_damage,
    normalized_to_xyxy,
    xyxy_to_normalized,
)


# ---------------------------------------------------------------------------
# Coordinate Conversion Tests
# ---------------------------------------------------------------------------


def test_xyxy_to_normalized_standard() -> None:
    """Standard box conversion to normalized coordinates."""
    # 640x480 image; box from (100, 120) to (300, 360)
    # w = 200, h = 240, cx = 200, cy = 240
    norm = xyxy_to_normalized([100, 120, 300, 360], img_w=640, img_h=480)
    assert len(norm) == 4
    assert pytest.approx(norm[0], rel=1e-3) == 200.0 / 640.0
    assert pytest.approx(norm[1], rel=1e-3) == 240.0 / 480.0
    assert pytest.approx(norm[2], rel=1e-3) == 200.0 / 640.0
    assert pytest.approx(norm[3], rel=1e-3) == 240.0 / 480.0


def test_xyxy_to_normalized_clamping() -> None:
    """Coordinates out of bounds should be clamped into [0, img_dim]."""
    norm = xyxy_to_normalized([-10, -50, 700, 800], img_w=640, img_h=480)
    assert norm[0] == 0.5  # cx is center
    assert norm[1] == 0.5  # cy is center
    assert norm[2] == 1.0  # full width
    assert norm[3] == 1.0  # full height


def test_xyxy_to_normalized_invalid_dims() -> None:
    """Non-positive image dimensions should raise ValueError."""
    with pytest.raises(ValueError, match="Invalid image dimensions"):
        xyxy_to_normalized([10, 10, 50, 50], img_w=0, img_h=480)

    with pytest.raises(ValueError, match="Invalid image dimensions"):
        xyxy_to_normalized([10, 10, 50, 50], img_w=640, img_h=-10)


def test_normalized_to_xyxy_standard() -> None:
    """Standard normalized box conversion to pixel coordinates."""
    box_xyxy = normalized_to_xyxy([0.5, 0.5, 0.5, 0.5], img_w=400, img_h=200)
    # cx=200, cy=100, w=200, h=100 -> x1=100, y1=50, x2=300, y2=150
    assert pytest.approx(box_xyxy[0]) == 100.0
    assert pytest.approx(box_xyxy[1]) == 50.0
    assert pytest.approx(box_xyxy[2]) == 300.0
    assert pytest.approx(box_xyxy[3]) == 150.0


def test_roundtrip_coordinate_conversion() -> None:
    """Converting xyxy -> norm -> xyxy recovers the original box."""
    orig_xyxy = [120.0, 85.0, 450.0, 390.0]
    img_w, img_h = 800, 600

    norm = xyxy_to_normalized(orig_xyxy, img_w, img_h)
    recov = normalized_to_xyxy(norm, img_w, img_h)

    for o, r in zip(orig_xyxy, recov):
        assert pytest.approx(o, abs=1e-4) == r


# ---------------------------------------------------------------------------
# Dataclass Tests
# ---------------------------------------------------------------------------


def test_damage_detection_dataclass_valid() -> None:
    """DamageDetection dataclass instantiates properly with valid inputs."""
    det = DamageDetection(
        box_xyxy=[10.0, 20.0, 100.0, 150.0],
        box_normalized=[0.2, 0.3, 0.4, 0.5],
        confidence=0.885,
        class_id=0,
        class_name="damage",
    )
    assert det.confidence == 0.885
    assert det.class_id == 0
    assert det.class_name == "damage"


def test_damage_detection_invalid_confidence() -> None:
    """Confidence outside [0, 1] raises ValueError."""
    with pytest.raises(ValueError, match="Confidence must be in"):
        DamageDetection(
            box_xyxy=[0, 0, 10, 10],
            box_normalized=[0, 0, 0.1, 0.1],
            confidence=1.2,
        )

    with pytest.raises(ValueError, match="Confidence must be in"):
        DamageDetection(
            box_xyxy=[0, 0, 10, 10],
            box_normalized=[0, 0, 0.1, 0.1],
            confidence=-0.05,
        )


def test_damage_detection_invalid_box_length() -> None:
    """Boxes with lengths != 4 raise ValueError."""
    with pytest.raises(ValueError, match="box_xyxy must have length 4"):
        DamageDetection(
            box_xyxy=[0, 0, 10],
            box_normalized=[0, 0, 0.1, 0.1],
            confidence=0.5,
        )

    with pytest.raises(ValueError, match="box_normalized must have length 4"):
        DamageDetection(
            box_xyxy=[0, 0, 10, 10],
            box_normalized=[0, 0, 0.1],
            confidence=0.5,
        )


def test_damage_detection_to_dict() -> None:
    """DamageDetection serializes cleanly to dict with correct keys and rounding."""
    det = DamageDetection(
        box_xyxy=[10.1234, 20.5678, 100.999, 150.444],
        box_normalized=[0.11111, 0.22222, 0.33333, 0.44444],
        confidence=0.887654,
    )
    d = det.to_dict()
    assert d["class_id"] == 0
    assert d["class_name"] == "damage"
    assert d["confidence"] == 0.8877
    assert d["box_xyxy"] == [10.12, 20.57, 101.0, 150.44]
    assert d["box_normalized"] == [0.1111, 0.2222, 0.3333, 0.4444]


# ---------------------------------------------------------------------------
# DamageDetector Image Loading & Validation Tests
# ---------------------------------------------------------------------------


@patch("ultralytics.YOLO")
def test_damage_detector_load_nonexistent_image(mock_yolo: MagicMock) -> None:
    """Non-existent image path raises FileNotFoundError."""
    detector = DamageDetector(model_path="dummy.pt")
    with pytest.raises(FileNotFoundError, match="Image not found at"):
        detector.predict("non_existent_car_photo.jpg")


@patch("ultralytics.YOLO")
def test_damage_detector_load_invalid_array(mock_yolo: MagicMock) -> None:
    """1D or 4D numpy array raises ValueError."""
    detector = DamageDetector(model_path="dummy.pt")
    with pytest.raises(ValueError, match="Expected 2D or 3D numpy image array"):
        detector.predict(np.zeros((100,)))

    with pytest.raises(ValueError, match="Expected 2D or 3D numpy image array"):
        detector.predict(np.zeros((1, 100, 100, 3)))


@patch("ultralytics.YOLO")
def test_damage_detector_load_grayscale_converts_to_bgr(mock_yolo: MagicMock) -> None:
    """2D grayscale array is automatically converted to 3-channel BGR."""
    detector = DamageDetector(model_path="dummy.pt")
    gray = np.zeros((100, 100), dtype=np.uint8)
    bgr = detector._load_image(gray)
    assert bgr.shape == (100, 100, 3)


# ---------------------------------------------------------------------------
# DamageDetector Inference & Mock Tests
# ---------------------------------------------------------------------------


def _create_mock_box(xyxy: list[float], conf: float, cls_id: int) -> MagicMock:
    box = MagicMock()
    box.xyxy = [torch.tensor(xyxy)]
    box.conf = [torch.tensor(conf)]
    box.cls = [torch.tensor(cls_id)]
    return box


@patch("ultralytics.YOLO")
def test_damage_detector_predict_with_detections(mock_yolo_cls: MagicMock) -> None:
    """DamageDetector parses model output boxes properly."""
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    mock_box = _create_mock_box([50.0, 60.0, 200.0, 250.0], conf=0.87, cls_id=0)
    mock_res = MagicMock()
    mock_res.boxes = [mock_box]
    mock_res.names = {0: "damage"}
    mock_model.predict.return_value = [mock_res]

    detector = DamageDetector(model_path="dummy.pt", conf_threshold=0.3)
    sample_img = np.zeros((480, 640, 3), dtype=np.uint8)

    dets = detector.predict(sample_img)
    assert len(dets) == 1
    assert dets[0].class_name == "damage"
    assert dets[0].class_id == 0
    assert pytest.approx(dets[0].confidence, rel=1e-2) == 0.87
    assert dets[0].box_xyxy == [50.0, 60.0, 200.0, 250.0]


@patch("ultralytics.YOLO")
def test_damage_detector_predict_empty_no_detections(mock_yolo_cls: MagicMock) -> None:
    """DamageDetector cleanly returns empty list when no damage is detected."""
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    mock_res = MagicMock()
    mock_res.boxes = []
    mock_model.predict.return_value = [mock_res]

    detector = DamageDetector(model_path="dummy.pt")
    sample_img = np.zeros((480, 640, 3), dtype=np.uint8)

    dets = detector.predict(sample_img)
    assert isinstance(dets, list)
    assert len(dets) == 0


@patch("ultralytics.YOLO")
def test_damage_detector_overlay_drawing(mock_yolo_cls: MagicMock) -> None:
    """predict_with_overlay draws bounding box and leaves original intact."""
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    mock_box = _create_mock_box([20.0, 20.0, 80.0, 80.0], conf=0.92, cls_id=0)
    mock_res = MagicMock()
    mock_res.boxes = [mock_box]
    mock_res.names = {0: "damage"}
    mock_model.predict.return_value = [mock_res]

    detector = DamageDetector(model_path="dummy.pt")
    original_img = np.zeros((200, 200, 3), dtype=np.uint8)
    original_copy = original_img.copy()

    dets, overlay = detector.predict_with_overlay(original_img)
    assert len(dets) == 1
    assert overlay.shape == original_img.shape
    # Original image must remain all zeros (unmodified)
    np.testing.assert_array_equal(original_img, original_copy)
    # Overlay should contain non-zero pixels from the drawn box
    assert np.any(overlay > 0)


@patch("ultralytics.YOLO")
def test_detect_damage_functional_api(mock_yolo_cls: MagicMock) -> None:
    """detect_damage convenience function invokes DamageDetector correctly."""
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    mock_box = _create_mock_box([10.0, 10.0, 50.0, 50.0], conf=0.75, cls_id=0)
    mock_res = MagicMock()
    mock_res.boxes = [mock_box]
    mock_res.names = {0: "damage"}
    mock_model.predict.return_value = [mock_res]

    sample_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dets = detect_damage(sample_img, model_path="dummy.pt", conf_threshold=0.5)

    assert len(dets) == 1
    assert dets[0].confidence == 0.75


@patch("ultralytics.YOLO")
def test_damage_detector_measure_cpu_latency(mock_yolo_cls: MagicMock) -> None:
    """measure_cpu_latency executes runs and returns positive latency in ms."""
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    mock_res = MagicMock()
    mock_res.boxes = []
    mock_model.predict.return_value = [mock_res]

    detector = DamageDetector(model_path="dummy.pt")
    sample_img = np.zeros((100, 100, 3), dtype=np.uint8)

    latency_ms = detector.measure_cpu_latency(sample_img, num_runs=5)
    assert isinstance(latency_ms, float)
    assert latency_ms >= 0.0
