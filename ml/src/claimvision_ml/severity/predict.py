"""Inference runtime and prediction interfaces for vehicle damage severity."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torchvision import transforms

from claimvision_ml.severity.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SEVERITY_CLASSES,
    SEVERITY_ID_TO_CLASS,
)
from claimvision_ml.severity.mobilenet import (
    SeverityMobileNetV2,
    load_severity_checkpoint,
)

logger = logging.getLogger(__name__)


@dataclass
class SeverityResult:
    """Standard assessment output for damage severity classification.

    Attributes:
        predicted_class: Predicted damage tier ("minor", "moderate", "severe").
        predicted_id: Integer class ID (0: minor, 1: moderate, 2: severe).
        confidence: Probability of the predicted class (0.0 to 1.0).
        probabilities: Softmax probability distribution over all classes.
        model_version: Model identifier (e.g. "SEV-MNV2-001").
        inference_time_ms: Wall-clock forward pass time in milliseconds.
        warnings: Descriptive warning flags (e.g. low confidence, low resolution).
    """

    predicted_class: str
    predicted_id: int
    confidence: float
    probabilities: dict[str, float]
    model_version: str = "SEV-MNV2-001"
    inference_time_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary."""
        return asdict(self)


_cached_models: dict[str, SeverityMobileNetV2] = {}


def _get_or_load_model(
    model_or_path: str | Path | SeverityMobileNetV2,
    device: str | torch.device = "cpu",
) -> SeverityMobileNetV2:
    """Retrieve an already-instantiated model or load it from disk with caching."""
    if isinstance(model_or_path, SeverityMobileNetV2):
        return model_or_path

    path_key = str(Path(model_or_path).resolve())
    if path_key in _cached_models:
        return _cached_models[path_key]

    model, _ = load_severity_checkpoint(path_key, device=device)
    _cached_models[path_key] = model
    return model


def predict_severity(
    image_input: str | Path | np.ndarray | Image.Image,
    model_or_path: str | Path | SeverityMobileNetV2,
    device: str | torch.device = "cpu",
    img_size: int = 224,
    confidence_warning_threshold: float = 0.50,
) -> SeverityResult:
    """Run end-to-end severity inference on a single image.

    Args:
        image_input: Path to image file, numpy array (BGR or RGB), or PIL Image.
        model_or_path: Instantiated SeverityMobileNetV2 or path to .pt checkpoint.
        device: 'cpu' or 'cuda'.
        img_size: Target image dimension (default 224).
        confidence_warning_threshold: Flag warning if top class confidence is below this.

    Returns:
        SeverityResult dataclass.
    """
    warnings: list[str] = []
    t0 = time.perf_counter()

    # 1. Image loading & decoding
    orig_w, orig_h = 0, 0
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.is_file():
            raise FileNotFoundError(f"Image not found: {image_input}")
        img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"OpenCV failed to decode image: {image_input}")
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
    elif isinstance(image_input, np.ndarray):
        orig_h, orig_w = image_input.shape[:2]
        if len(image_input.shape) == 2:
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 3:
            # Assume RGB if passed as ndarray
            img_rgb = image_input
        else:
            raise ValueError(f"Unsupported image array shape: {image_input.shape}")
        img_pil = Image.fromarray(img_rgb)
    elif isinstance(image_input, Image.Image):
        orig_w, orig_h = image_input.size
        img_pil = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    if orig_w < 224 or orig_h < 224:
        warnings.append(f"Low resolution input ({orig_w}x{orig_h}); minimum recommended is 224x224")

    # 2. Preprocessing
    resize_dim = int(img_size * 256 / 224)
    transform = transforms.Compose(
        [
            transforms.Resize(resize_dim),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # 3. Model inference
    model = _get_or_load_model(model_or_path, device=device)
    model.eval()

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    pred_id = int(np.argmax(probs))
    pred_class = SEVERITY_ID_TO_CLASS[pred_id]
    conf = float(probs[pred_id])

    if conf < confidence_warning_threshold:
        warnings.append(
            f"Low confidence ({conf:.2%}); damage pattern may be ambiguous between {SEVERITY_CLASSES}"
        )

    prob_dict = {
        SEVERITY_ID_TO_CLASS[i]: round(float(probs[i]), 4)
        for i in range(len(SEVERITY_CLASSES))
    }

    return SeverityResult(
        predicted_class=pred_class,
        predicted_id=pred_id,
        confidence=round(conf, 4),
        probabilities=prob_dict,
        model_version="SEV-MNV2-001",
        inference_time_ms=round(latency_ms, 2),
        warnings=warnings,
    )
