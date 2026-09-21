"""test_part_detector.py — Unit tests for Damaged-Part Detector (DET-PART-001).

Phase: 10
Task: DET-PART-001
"""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

from claimvision_ml.detection.parts import (
    DEFAULT_PART_COLORS,
    PARTS_CLASS_MAP,
    PARTS_CLASS_NAMES,
    PartDetection,
    PartDetector,
    detect_parts,
)


# ---------------------------------------------------------------------------
# Canonical Constants Tests
# ---------------------------------------------------------------------------


def test_canonical_constants() -> None:
    """Validate that canonical constants have exactly 5 classes in expected order."""
    assert len(PARTS_CLASS_NAMES) == 5
    assert PARTS_CLASS_NAMES == ["headlamp", "front_bumper", "hood", "door", "rear_bumper"]
    assert len(PARTS_CLASS_MAP) == 5
    for i, name in enumerate(PARTS_CLASS_NAMES):
        assert PARTS_CLASS_MAP[i] == name
        assert name in DEFAULT_PART_COLORS


# ---------------------------------------------------------------------------
# Dataclass Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("class_id,class_name", [
    (0, "headlamp"),
    (1, "front_bumper"),
    (2, "hood"),
    (3, "door"),
    (4, "rear_bumper"),
])
def test_part_detection_valid_all_classes(class_id: int, class_name: str) -> None:
    """PartDetection creates cleanly for all 5 defined vehicle parts."""
    det = PartDetection(
        box_xyxy=[10.0, 20.0, 100.0, 150.0],
        box_normalized=[0.2, 0.3, 0.4, 0.5],
        confidence=0.88,
        class_id=class_id,
        class_name=class_name,
    )
    assert det.class_id == class_id
    assert det.class_name == class_name
    assert det.confidence == 0.88


def test_part_detection_invalid_class_id() -> None:
    """Invalid class_id raises ValueError."""
    with pytest.raises(ValueError, match="Invalid class_id"):
        PartDetection(
            box_xyxy=[0, 0, 10, 10],
            box_normalized=[0, 0, 0.1, 0.1],
            confidence=0.9,
            class_id=5,
            class_name="headlamp",
        )


def test_part_detection_invalid_class_name() -> None:
    """Invalid class_name raises ValueError."""
    with pytest.raises(ValueError, match="Invalid class_name"):
        PartDetection(
            box_xyxy=[0, 0, 10, 10],
            box_normalized=[0, 0, 0.1, 0.1],
            confidence=0.9,
            class_id=0,
            class_name="windshield",
        )


def test_part_detection_invalid_confidence() -> None:
    """Confidence outside [0, 1] raises ValueError."""
    with pytest.raises(ValueError, match="Confidence must be in"):
        PartDetection(
            box_xyxy=[0, 0, 10, 10],
            box_normalized=[0, 0, 0.1, 0.1],
            confidence=1.5,
            class_id=0,
            class_name="headlamp",
        )


def test_part_detection_invalid_box_len() -> None:
    """Box lengths != 4 raise ValueError."""
    with pytest.raises(ValueError, match="box_xyxy must have length 4"):
        PartDetection(
            box_xyxy=[0, 0, 10],
            box_normalized=[0, 0, 0.1, 0.1],
            confidence=0.5,
            class_id=0,
            class_name="headlamp",
        )


def test_part_detection_to_dict() -> None:
    """PartDetection serializes cleanly with proper rounding."""
    det = PartDetection(
        box_xyxy=[10.1234, 20.5678, 100.999, 150.444],
        box_normalized=[0.11111, 0.22222, 0.33333, 0.44444],
        confidence=0.887654,
        class_id=1,
        class_name="front_bumper",
    )
    d = det.to_dict()
    assert d["class_id"] == 1
    assert d["class_name"] == "front_bumper"
    assert d["confidence"] == 0.8877
    assert d["box_xyxy"] == [10.12, 20.57, 101.0, 150.44]
    assert d["box_normalized"] == [0.1111, 0.2222, 0.3333, 0.4444]


# ---------------------------------------------------------------------------
# PartDetector Image Loading & Validation Tests
# ---------------------------------------------------------------------------


@patch("ultralytics.YOLO")
def test_part_detector_load_nonexistent_image(mock_yolo: MagicMock) -> None:
    """Non-existent image path raises FileNotFoundError."""
    detector = PartDetector(model_path="dummy.pt")
    with pytest.raises(FileNotFoundError, match="Image not found at"):
        detector.predict("non_existent_part_photo.jpg")


@patch("ultralytics.YOLO")
def test_part_detector_load_invalid_array(mock_yolo: MagicMock) -> None:
    """1D or 4D numpy array raises ValueError."""
    detector = PartDetector(model_path="dummy.pt")
    with pytest.raises(ValueError, match="Expected 2D or 3D numpy image array"):
        detector.predict(np.zeros((100,)))


@patch("ultralytics.YOLO")
def test_part_detector_load_grayscale_converts_to_bgr(mock_yolo: MagicMock) -> None:
    """2D grayscale image automatically converts to 3D BGR."""
    detector = PartDetector(model_path="dummy.pt")
    gray = np.zeros((100, 100), dtype=np.uint8)
    bgr = detector._load_image(gray)
    assert bgr.shape == (100, 100, 3)


# ---------------------------------------------------------------------------
# PartDetector Inference & Mock Tests
# ---------------------------------------------------------------------------


def _create_mock_box(xyxy: list[float], conf: float, cls_id: int) -> MagicMock:
    box = MagicMock()
    box.xyxy = [torch.tensor(xyxy)]
    box.conf = [torch.tensor(conf)]
    box.cls = [torch.tensor(cls_id)]
    return box


@patch("ultralytics.YOLO")
def test_part_detector_predict_multi_part(mock_yolo_cls: MagicMock) -> None:
    """PartDetector correctly maps multiple bounding boxes to their respective part names."""
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    box1 = _create_mock_box([10.0, 10.0, 50.0, 50.0], conf=0.85, cls_id=0)  # headlamp
    box2 = _create_mock_box([30.0, 80.0, 200.0, 120.0], conf=0.92, cls_id=1) # front_bumper

    mock_res = MagicMock()
    mock_res.boxes = [box1, box2]
    mock_res.names = PARTS_CLASS_MAP
    mock_model.predict.return_value = [mock_res]

    detector = PartDetector(model_path="dummy.pt", conf_threshold=0.25)
    sample_img = np.zeros((480, 640, 3), dtype=np.uint8)

    dets = detector.predict(sample_img)
    assert len(dets) == 2
    assert dets[0].class_name == "headlamp"
    assert dets[0].class_id == 0
    assert pytest.approx(dets[0].confidence, rel=1e-2) == 0.85
    assert dets[1].class_name == "front_bumper"
    assert dets[1].class_id == 1
    assert pytest.approx(dets[1].confidence, rel=1e-2) == 0.92


@patch("ultralytics.YOLO")
def test_part_detector_predict_empty_boxes(mock_yolo_cls: MagicMock) -> None:
    """PartDetector cleanly returns empty list when no parts are detected."""
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    mock_res = MagicMock()
    mock_res.boxes = []
    mock_model.predict.return_value = [mock_res]

    detector = PartDetector(model_path="dummy.pt")
    sample_img = np.zeros((480, 640, 3), dtype=np.uint8)

    dets = detector.predict(sample_img)
    assert isinstance(dets, list)
    assert len(dets) == 0


@patch("ultralytics.YOLO")
def test_part_detector_overlay_drawing(mock_yolo_cls: MagicMock) -> None:
    """predict_with_overlay draws colored boxes and leaves original unmodified."""
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    box = _create_mock_box([20.0, 20.0, 80.0, 80.0], conf=0.88, cls_id=2)  # hood
    mock_res = MagicMock()
    mock_res.boxes = [box]
    mock_res.names = PARTS_CLASS_MAP
    mock_model.predict.return_value = [mock_res]

    detector = PartDetector(model_path="dummy.pt")
    original_img = np.zeros((200, 200, 3), dtype=np.uint8)
    original_copy = original_img.copy()

    dets, overlay = detector.predict_with_overlay(original_img)
    assert len(dets) == 1
    assert dets[0].class_name == "hood"
    # Original image remains pristine
    np.testing.assert_array_equal(original_img, original_copy)
    # Overlay has drawn pixels
    assert np.any(overlay > 0)


@patch("ultralytics.YOLO")
def test_part_detector_get_detected_part_names(mock_yolo_cls: MagicMock) -> None:
    """get_detected_part_names returns sorted unique list of part names."""
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    # Two headlamps and one hood
    box1 = _create_mock_box([10, 10, 40, 40], conf=0.8, cls_id=0)
    box2 = _create_mock_box([50, 10, 80, 40], conf=0.9, cls_id=0)
    box3 = _create_mock_box([20, 50, 70, 90], conf=0.75, cls_id=2)

    mock_res = MagicMock()
    mock_res.boxes = [box1, box2, box3]
    mock_res.names = PARTS_CLASS_MAP
    mock_model.predict.return_value = [mock_res]

    detector = PartDetector(model_path="dummy.pt")
    sample_img = np.zeros((100, 100, 3), dtype=np.uint8)

    part_names = detector.get_detected_part_names(sample_img)
    assert part_names == ["headlamp", "hood"]


@patch("ultralytics.YOLO")
def test_detect_parts_functional_api(mock_yolo_cls: MagicMock) -> None:
    """detect_parts functional API invokes PartDetector correctly."""
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    box = _create_mock_box([10, 10, 50, 50], conf=0.91, cls_id=3) # door
    mock_res = MagicMock()
    mock_res.boxes = [box]
    mock_res.names = PARTS_CLASS_MAP
    mock_model.predict.return_value = [mock_res]

    sample_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dets = detect_parts(sample_img, model_path="dummy.pt", conf_threshold=0.5)

    assert len(dets) == 1
    assert dets[0].class_name == "door"
    assert pytest.approx(dets[0].confidence, rel=1e-2) == 0.91


@patch("ultralytics.YOLO")
def test_part_detector_measure_cpu_latency(mock_yolo_cls: MagicMock) -> None:
    """measure_cpu_latency benchmarks runs and returns positive latency in ms."""
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    mock_res = MagicMock()
    mock_res.boxes = []
    mock_model.predict.return_value = [mock_res]

    detector = PartDetector(model_path="dummy.pt")
    sample_img = np.zeros((100, 100, 3), dtype=np.uint8)

    lat = detector.measure_cpu_latency(sample_img, num_runs=3)
    assert isinstance(lat, float)
    assert lat >= 0.0
