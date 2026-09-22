"""location/inference.py — Unified runtime inference for location classifiers.

Task IDs : LOC-MNV2-001, LOC-EFF-001
Phase    : 11a / 11b
Owner    : Detection ML member / Antigravity

This module provides:
- LocationClassification  — dataclass holding class, confidence, top-3
- LocationClassifier      — runtime wrapper that loads either MobileNetV2 or
                            EfficientNet-B0 checkpoint and runs inference
- classify_location       — functional API
- export_location_onnx    — ONNX export helper

Per AGENTS.md §12 (API contract): model loading must happen at initialisation
(singleton), not per-request.  Inference returns model_type, confidence,
warnings, and timing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F

from claimvision_ml.location.dataset import (
    LOCATION_CLASSES,
    LOCATION_CLASS_TO_ID,
    LOCATION_ID_TO_CLASS,
    LOCATION_IMAGE_SIZE,
    get_location_transforms,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LocationClassification:
    """Result of a single location inference call.

    Attributes
    ----------
    class_id:
        Predicted class integer (0–4).
    class_name:
        Predicted class name (e.g. ``"front_bumper"``).
    confidence:
        Softmax probability of the predicted class in [0, 1].
    top3:
        List of up to 3 ``(class_name, probability)`` tuples ordered by
        decreasing probability.
    model_type:
        ``"mobilenet"`` or ``"efficientnet"``.
    latency_ms:
        Wall-clock inference time in milliseconds.
    warning:
        Non-empty string when confidence is below a reliability threshold.
    """

    class_id: int
    class_name: str
    confidence: float
    top3: list[tuple[str, float]] = field(default_factory=list)
    model_type: str = "mobilenet"
    latency_ms: float = 0.0
    warning: str = ""

    def __post_init__(self) -> None:
        if self.class_id not in LOCATION_CLASS_TO_ID.values():
            raise ValueError(
                f"class_id must be in 0–{len(LOCATION_CLASSES) - 1}, got {self.class_id}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )
        if self.class_name not in LOCATION_CLASSES:
            raise ValueError(
                f"class_name must be one of {LOCATION_CLASSES}, got {self.class_name!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a clean JSON-safe dictionary."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "top3": [(n, round(float(p), 4)) for n, p in self.top3],
            "model_type": self.model_type,
            "latency_ms": round(self.latency_ms, 2),
            "warning": self.warning,
        }


# ---------------------------------------------------------------------------
# Runtime wrapper
# ---------------------------------------------------------------------------

LOW_CONFIDENCE_THRESHOLD: float = 0.40  # below this → include warning


class LocationClassifier:
    """Runtime location classifier. Loads one checkpoint and runs inference.

    Parameters
    ----------
    model_path:
        Path to a ``.pt`` state-dict checkpoint.
    model_type:
        ``"mobilenet"`` or ``"efficientnet"``.
    num_classes:
        Number of output classes (default 5).
    dropout:
        Dropout used during training (must match checkpoint).
    device:
        Inference device (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        model_path: str | Path,
        model_type: Literal["mobilenet", "efficientnet"] = "mobilenet",
        num_classes: int = 5,
        dropout: float = 0.3,
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = torch.device(device)
        self._model: torch.nn.Module = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        if self.model_type == "mobilenet":
            from claimvision_ml.location.mobilenet import load_location_mobilenet
            model = load_location_mobilenet(
                self.model_path, num_classes=self.num_classes,
                dropout=self.dropout, device=str(self.device),
            )
        elif self.model_type == "efficientnet":
            from claimvision_ml.location.efficientnet import load_location_efficientnet
            model = load_location_efficientnet(
                self.model_path, num_classes=self.num_classes,
                dropout=self.dropout, device=str(self.device),
            )
        else:
            raise ValueError(
                f"model_type must be 'mobilenet' or 'efficientnet', got {self.model_type!r}"
            )
        model.to(self.device)
        model.eval()
        return model

    def _preprocess(self, image: Any) -> torch.Tensor:
        """Convert image to a normalised tensor (1, 3, 224, 224)."""
        from PIL import Image as PILImage
        import numpy as np

        transform = get_location_transforms("val")

        if isinstance(image, torch.Tensor):
            tensor = image.unsqueeze(0) if image.dim() == 3 else image
        elif isinstance(image, np.ndarray):
            pil = PILImage.fromarray(image.astype("uint8"))
            tensor = transform(pil).unsqueeze(0)
        elif isinstance(image, PILImage.Image):
            tensor = transform(image).unsqueeze(0)
        elif isinstance(image, (str, Path)):
            pil = PILImage.open(image).convert("RGB")
            tensor = transform(pil).unsqueeze(0)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        return tensor.to(self.device)

    def predict(self, image: Any) -> LocationClassification:
        """Run inference on a single image.

        Parameters
        ----------
        image:
            File path, PIL Image, numpy BGR/RGB array, or a pre-processed tensor.

        Returns
        -------
        :class:`LocationClassification` result.
        """
        tensor = self._preprocess(image)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self._model(tensor)  # (1, 5)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()  # list[float] len 5
        pred_id = int(torch.argmax(torch.tensor(probs)).item())
        pred_name = LOCATION_ID_TO_CLASS[pred_id]
        confidence = float(probs[pred_id])

        # Top-3
        sorted_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        top3 = [(LOCATION_ID_TO_CLASS[i], float(probs[i])) for i in sorted_idx[:3]]

        warning = ""
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            warning = (
                f"Low confidence ({confidence:.1%}). "
                "Prediction is uncertain — consider using YOLO bounding-box classification."
            )

        return LocationClassification(
            class_id=pred_id,
            class_name=pred_name,
            confidence=confidence,
            top3=top3,
            model_type=self.model_type,
            latency_ms=latency_ms,
            warning=warning,
        )


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------

def classify_location(
    image: Any,
    model: torch.nn.Module,
    model_type: str = "mobilenet",
    device: str = "cpu",
) -> LocationClassification:
    """Functional location classification API.

    Parameters
    ----------
    image:
        File path, PIL Image, numpy array, or pre-processed tensor.
    model:
        A loaded :class:`LocationMobileNet` or :class:`LocationEfficientNet` in eval mode.
    model_type:
        ``"mobilenet"`` or ``"efficientnet"`` — used to populate the result.
    device:
        Inference device.

    Returns
    -------
    :class:`LocationClassification`.
    """
    from PIL import Image as PILImage
    import numpy as np

    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    transform = get_location_transforms("val")

    if isinstance(image, torch.Tensor):
        tensor = image.unsqueeze(0).to(dev) if image.dim() == 3 else image.to(dev)
    elif isinstance(image, np.ndarray):
        tensor = transform(PILImage.fromarray(image.astype("uint8"))).unsqueeze(0).to(dev)
    elif isinstance(image, PILImage.Image):
        tensor = transform(image).unsqueeze(0).to(dev)
    elif isinstance(image, (str, Path)):
        pil = PILImage.open(image).convert("RGB")
        tensor = transform(pil).unsqueeze(0).to(dev)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    pred_id = int(torch.tensor(probs).argmax().item())
    pred_name = LOCATION_ID_TO_CLASS[pred_id]
    confidence = float(probs[pred_id])
    sorted_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    top3 = [(LOCATION_ID_TO_CLASS[i], float(probs[i])) for i in sorted_idx[:3]]

    warning = ""
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        warning = (
            f"Low confidence ({confidence:.1%}). "
            "Prediction is uncertain — consider using YOLO classification."
        )

    return LocationClassification(
        class_id=pred_id,
        class_name=pred_name,
        confidence=confidence,
        top3=top3,
        model_type=model_type,
        latency_ms=latency_ms,
        warning=warning,
    )


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_location_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    image_size: int = LOCATION_IMAGE_SIZE,
    opset_version: int = 17,
) -> Path:
    """Export a location classifier to ONNX format.

    Parameters
    ----------
    model:
        Trained :class:`LocationMobileNet` or :class:`LocationEfficientNet`.
    output_path:
        Destination ``.onnx`` file path.
    image_size:
        Input image side length (default 224).
    opset_version:
        ONNX opset version (default 17).

    Returns
    -------
    :class:`Path` to the exported ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.cpu()
    dummy = torch.zeros(1, 3, image_size, image_size)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=opset_version,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
    return output_path
