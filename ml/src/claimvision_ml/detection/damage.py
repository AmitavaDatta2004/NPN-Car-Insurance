"""damage.py — Reusable Generic Damage Detection using YOLOv8.

Task ID : DET-YOLO-001
Phase   : 9
Owner   : Member 4 (Detection ML)

Public API:
    DamageDetection     — dataclass holding bounding box, class, and confidence
    DamageDetector      — high-level detector wrapper for inference & overlay
    detect_damage       — convenience functional API
    export_damage_onnx  — export YOLOv8 model to ONNX format
    xyxy_to_normalized  — convert [x1, y1, x2, y2] to [cx, cy, w, h] normalized
    normalized_to_xyxy  — convert [cx, cy, w, h] normalized to [x1, y1, x2, y2]
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Coordinate conversions
# ---------------------------------------------------------------------------


def xyxy_to_normalized(
    box_xyxy: list[float] | tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> list[float]:
    """Convert [x1, y1, x2, y2] pixel coordinates to normalized [cx, cy, w, h].

    All output values are in [0, 1]. Clamped to valid bounds.
    """
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid image dimensions: {img_w}x{img_h}")

    x1, y1, x2, y2 = box_xyxy
    x1 = max(0.0, min(float(x1), float(img_w)))
    y1 = max(0.0, min(float(y1), float(img_h)))
    x2 = max(0.0, min(float(x2), float(img_w)))
    y2 = max(0.0, min(float(y2), float(img_h)))

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0

    return [cx / img_w, cy / img_h, w / img_w, h / img_h]


def normalized_to_xyxy(
    box_norm: list[float] | tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> list[float]:
    """Convert normalized [cx, cy, w, h] in [0, 1] to pixel [x1, y1, x2, y2]."""
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid image dimensions: {img_w}x{img_h}")

    cx, cy, w, h = box_norm
    w_px = w * img_w
    h_px = h * img_h
    x1 = cx * img_w - w_px / 2.0
    y1 = cy * img_h - h_px / 2.0
    x2 = x1 + w_px
    y2 = y1 + h_px

    return [
        max(0.0, min(float(x1), float(img_w))),
        max(0.0, min(float(y1), float(img_h))),
        max(0.0, min(float(x2), float(img_w))),
        max(0.0, min(float(y2), float(img_h))),
    ]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DamageDetection:
    """Represents a single localized damage region."""

    box_xyxy: list[float]
    box_normalized: list[float]
    confidence: float
    class_id: int = 0
    class_name: str = "damage"

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if len(self.box_xyxy) != 4:
            raise ValueError(f"box_xyxy must have length 4, got {len(self.box_xyxy)}")
        if len(self.box_normalized) != 4:
            raise ValueError(
                f"box_normalized must have length 4, got {len(self.box_normalized)}"
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
# DamageDetector Class
# ---------------------------------------------------------------------------


class DamageDetector:
    """High-level damage detection wrapper powered by Ultralytics YOLO."""

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ) -> None:
        """Initialize the DamageDetector with a trained YOLO weights file.

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
    ) -> list[DamageDetection]:
        """Run damage detection on an input image.

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
        list[DamageDetection]:
            List of detected damage bounding boxes. Empty if no damage detected.
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

        detections: list[DamageDetection] = []
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
                r.names.get(cls_id, "damage")
                if hasattr(r, "names") and r.names
                else "damage"
            )

            box_norm = xyxy_to_normalized(xyxy, img_w, img_h)
            detections.append(
                DamageDetection(
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
        color: tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
        conf_threshold: float | None = None,
    ) -> tuple[list[DamageDetection], np.ndarray]:
        """Run detection and draw visual bounding boxes on an image copy.

        The original image is never modified.

        Parameters
        ----------
        image:
            File path or numpy BGR array.
        color:
            BGR color for bounding boxes (default: red `(0, 0, 255)`).
        thickness:
            Line thickness for boxes.
        conf_threshold:
            Optional override for confidence threshold.

        Returns
        -------
        tuple[list[DamageDetection], np.ndarray]:
            Tuple containing the list of detections and the annotated BGR image copy.
        """
        img_bgr = self._load_image(image)
        overlay = img_bgr.copy()

        detections = self.predict(img_bgr, conf_threshold=conf_threshold)

        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det.box_xyxy]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

            label = f"{det.class_name} {det.confidence * 100:.1f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thick = 1
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, font_thick
            )

            # Draw label background rectangle
            label_y1 = max(0, y1 - text_h - baseline - 4)
            label_y2 = label_y1 + text_h + baseline + 4
            cv2.rectangle(
                overlay,
                (x1, label_y1),
                (x1 + text_w + 4, label_y2),
                color,
                -1,
            )
            # Draw white text
            cv2.putText(
                overlay,
                label,
                (x1 + 2, label_y2 - baseline - 2),
                font,
                font_scale,
                (255, 255, 255),
                font_thick,
                cv2.LINE_AA,
            )

        return detections, overlay

    def measure_cpu_latency(
        self,
        sample_image: str | Path | np.ndarray,
        num_runs: int = 20,
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


def detect_damage(
    image: str | Path | np.ndarray,
    model_path: str | Path,
    conf_threshold: float = 0.25,
    device: str = "cpu",
) -> list[DamageDetection]:
    """Detect damaged regions on an image using a specified YOLO model.

    Convenience wrapper fulfilling the public detection package API.
    """
    detector = DamageDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        device=device,
    )
    return detector.predict(image)


def export_damage_onnx(
    model_path: str | Path,
    output_path: str | Path | None = None,
    imgsz: int = 640,
) -> Path:
    """Export a trained PyTorch YOLO model (.pt) to ONNX format.

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
