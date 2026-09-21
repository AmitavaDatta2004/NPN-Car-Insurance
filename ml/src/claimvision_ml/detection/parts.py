"""parts.py — Reusable Damaged-Part Detection using YOLOv8 (5 Classes).

Task ID : DET-PART-001
Phase   : 10
Owner   : Member 4 (Detection ML)

Classes (nc=5):
    0: headlamp
    1: front_bumper
    2: hood
    3: door
    4: rear_bumper

Public API:
    PartDetection          — dataclass holding bounding box, part class, and confidence
    PartDetector           — high-level detector wrapper for inference & multi-color overlays
    detect_parts           — convenience functional API
    export_parts_onnx      — export YOLOv8 model to ONNX format
    PARTS_CLASS_NAMES      — canonical list of 5 part class names
    DEFAULT_PART_COLORS    — distinct BGR colors for each vehicle part
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from claimvision_ml.detection.damage import (
    normalized_to_xyxy,
    xyxy_to_normalized,
)


PARTS_CLASS_NAMES: list[str] = [
    "headlamp",
    "front_bumper",
    "hood",
    "door",
    "rear_bumper",
]

PARTS_CLASS_MAP: dict[int, str] = {i: name for i, name in enumerate(PARTS_CLASS_NAMES)}

# Distinct BGR colors for clear judge visual inspection
DEFAULT_PART_COLORS: dict[str, tuple[int, int, int]] = {
    "headlamp": (0, 255, 255),      # Yellow
    "front_bumper": (255, 255, 0),  # Cyan
    "hood": (0, 255, 0),            # Green
    "door": (255, 0, 0),            # Blue
    "rear_bumper": (255, 0, 255),   # Magenta
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PartDetection:
    """Represents a single localized damaged vehicle component."""

    box_xyxy: list[float]
    box_normalized: list[float]
    confidence: float
    class_id: int
    class_name: str

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if len(self.box_xyxy) != 4:
            raise ValueError(f"box_xyxy must have length 4, got {len(self.box_xyxy)}")
        if len(self.box_normalized) != 4:
            raise ValueError(
                f"box_normalized must have length 4, got {len(self.box_normalized)}"
            )
        if self.class_id not in PARTS_CLASS_MAP:
            raise ValueError(
                f"Invalid class_id: {self.class_id}. Expected one of {list(PARTS_CLASS_MAP.keys())}"
            )
        if self.class_name not in PARTS_CLASS_NAMES:
            raise ValueError(
                f"Invalid class_name: {self.class_name}. Expected one of {PARTS_CLASS_NAMES}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize detection to a clean dictionary."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "box_xyxy": [round(float(v), 2) for v in self.box_xyxy],
            "box_normalized": [round(float(v), 4) for v in self.box_normalized],
        }


# ---------------------------------------------------------------------------
# PartDetector Class
# ---------------------------------------------------------------------------


class PartDetector:
    """High-level damaged part detection wrapper powered by Ultralytics YOLO."""

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ) -> None:
        """Initialize the PartDetector with a trained YOLO weights file.

        Parameters
        ----------
        model_path:
            Path to .pt weights file or YOLO model identifier (e.g. 'yolov8n.pt').
        conf_threshold:
            Confidence threshold to filter weak detections.
        iou_threshold:
            NMS IoU threshold.
        device:
            Inference device ('cpu', 'cuda', etc.).
        """
        self.model_path = Path(model_path) if isinstance(model_path, str) else model_path
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.device = str(device)
        self._model: Any = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the Ultralytics YOLO model."""
        from ultralytics import YOLO

        target = str(self.model_path)
        self._model = YOLO(target)

    @property
    def model(self) -> Any:
        """Return underlying YOLO model instance."""
        return self._model

    def _load_image(self, image: str | Path | np.ndarray) -> np.ndarray:
        """Load image into BGR numpy array if path provided, and validate."""
        if isinstance(image, (str, Path)):
            img_path = Path(image)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found at: {img_path}")
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Could not decode image at: {img_path}")
            return img
        elif isinstance(image, np.ndarray):
            if image.ndim not in (2, 3):
                raise ValueError(
                    f"Expected 2D or 3D numpy image array, got shape {image.shape}"
                )
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def predict(
        self,
        image: str | Path | np.ndarray,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> list[PartDetection]:
        """Run damaged part detection on an input vehicle photograph.

        Parameters
        ----------
        image:
            File path or numpy BGR array.
        conf_threshold:
            Optional override for confidence threshold.
        iou_threshold:
            Optional override for NMS IoU threshold.

        Returns
        -------
        list[PartDetection]:
            List of detected vehicle components. Empty if no components detected.
        """
        img_bgr = self._load_image(image)
        img_h, img_w = img_bgr.shape[:2]

        conf = self.conf_threshold if conf_threshold is None else float(conf_threshold)
        iou = self.iou_threshold if iou_threshold is None else float(iou_threshold)

        results = self._model.predict(
            source=img_bgr,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False,
        )

        detections: list[PartDetection] = []
        if not results:
            return detections

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        for box in r.boxes:
            xyxy = box.xyxy[0].tolist()
            confidence = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            class_name = (
                r.names.get(cls_id, PARTS_CLASS_MAP.get(cls_id, f"part_{cls_id}"))
                if hasattr(r, "names") and r.names
                else PARTS_CLASS_MAP.get(cls_id, f"part_{cls_id}")
            )

            box_norm = xyxy_to_normalized(xyxy, img_w, img_h)
            detections.append(
                PartDetection(
                    box_xyxy=[float(v) for v in xyxy],
                    box_normalized=box_norm,
                    confidence=confidence,
                    class_id=cls_id,
                    class_name=class_name,
                )
            )

        return detections

    def predict_with_overlay(
        self,
        image: str | Path | np.ndarray,
        color_map: dict[str, tuple[int, int, int]] | None = None,
        thickness: int = 2,
        conf_threshold: float | None = None,
    ) -> tuple[list[PartDetection], np.ndarray]:
        """Run detection and draw multi-color visual bounding boxes on an image copy.

        The original image is never modified.

        Parameters
        ----------
        image:
            File path or numpy BGR array.
        color_map:
            Optional dictionary mapping class name to BGR color tuple.
        thickness:
            Line thickness for boxes.
        conf_threshold:
            Optional override for confidence threshold.

        Returns
        -------
        tuple[list[PartDetection], np.ndarray]:
            Tuple containing the list of detections and the annotated BGR image copy.
        """
        img_bgr = self._load_image(image)
        overlay = img_bgr.copy()

        palette = DEFAULT_PART_COLORS if color_map is None else color_map
        detections = self.predict(img_bgr, conf_threshold=conf_threshold)

        for det in detections:
            color = palette.get(det.class_name, (0, 255, 0))
            x1, y1, x2, y2 = [int(round(v)) for v in det.box_xyxy]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

            label = f"{det.class_name} {det.confidence * 100:.0f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thick = 1
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, font_thick
            )

            label_y1 = max(0, y1 - text_h - baseline - 4)
            label_y2 = label_y1 + text_h + baseline + 4
            cv2.rectangle(
                overlay,
                (x1, label_y1),
                (x1 + text_w + 4, label_y2),
                color,
                -1,
            )
            # Text in black or white depending on luminance of label background
            luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
            text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)

            cv2.putText(
                overlay,
                label,
                (x1 + 2, label_y2 - baseline - 2),
                font,
                font_scale,
                text_color,
                font_thick,
                cv2.LINE_AA,
            )

        return detections, overlay

    def get_detected_part_names(
        self,
        image: str | Path | np.ndarray,
        conf_threshold: float | None = None,
    ) -> list[str]:
        """Return unique list of detected part names for downstream repair costing."""
        detections = self.predict(image, conf_threshold=conf_threshold)
        unique_parts = sorted(list({det.class_name for det in detections}))
        return unique_parts

    def measure_cpu_latency(
        self,
        sample_image: str | Path | np.ndarray,
        num_runs: int = 10,
    ) -> float:
        """Benchmark average CPU inference latency in milliseconds per image."""
        img_bgr = self._load_image(sample_image)

        # Warmup
        _ = self.predict(img_bgr)

        start = time.perf_counter()
        for _ in range(num_runs):
            _ = self.predict(img_bgr)
        elapsed = time.perf_counter() - start

        return (elapsed / num_runs) * 1000.0


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def detect_parts(
    image: str | Path | np.ndarray,
    model_path: str | Path,
    conf_threshold: float = 0.25,
    device: str = "cpu",
) -> list[PartDetection]:
    """Detect damaged vehicle parts using a specified YOLO model.

    Convenience wrapper fulfilling the public detection package API.
    """
    detector = PartDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        device=device,
    )
    return detector.predict(image)


def export_parts_onnx(
    model_path: str | Path,
    output_path: str | Path | None = None,
    imgsz: int = 640,
) -> Path:
    """Export a trained PyTorch YOLO part model (.pt) to ONNX format.

    Parameters
    ----------
    model_path:
        Path to .pt model weights.
    output_path:
        Optional target destination. If omitted, uses default Ultralytics export path.
    imgsz:
        Inference image size (default: 640).

    Returns
    -------
    Path:
        Path to the exported .onnx model file.
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    exported_file = model.export(format="onnx", imgsz=imgsz)
    exported_path = Path(exported_file)

    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if exported_path.resolve() != target.resolve():
            import shutil

            shutil.copy2(exported_path, target)
        return target

    return exported_path
