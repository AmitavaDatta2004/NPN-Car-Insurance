"""Runtime fraud-risk prediction for a single image.

This module provides the primary public function predict_fraud(), which
is called by the FastAPI backend and the unified inference pipeline.

Output contract (FraudResult):
    - fraud_probability: float [0.0, 1.0] — sigmoid of the raw logit
    - risk_level: "low" | "medium" | "high" based on frozen thresholds
    - route: "CONTINUE" | "FRAUD_REVIEW"
    - threshold_version, model_version, inference_ms, warnings

Important: A high risk_level means the image has suspicious visual
characteristics relative to training data. It does NOT confirm
insurance fraud and must not be used to automatically reject a claim.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image

from claimvision_ml.fraud.dataset import get_transforms
from claimvision_ml.fraud.model import load_fraud_model


@dataclass
class FraudResult:
    """Output of predict_fraud() for a single image.

    Attributes:
        image_path: Absolute or relative path to the evaluated image.
        fraud_probability: Sigmoid probability [0.0, 1.0]. Higher = more suspicious.
        risk_level: "low", "medium", or "high" based on threshold_version thresholds.
        route: Downstream routing decision.
            "CONTINUE"    -> proceed to damage/severity assessment.
            "FRAUD_REVIEW" -> stop automatic assessment; route to human fraud review.
        threshold_version: Version string of the threshold config file used.
        model_version: Experiment ID of the loaded model.
        inference_ms: Wall-clock inference time in milliseconds.
        warnings: Non-fatal issues encountered during inference (e.g. resized image).
    """

    image_path: str
    fraud_probability: float
    risk_level: str        # "low" | "medium" | "high"
    route: str             # "CONTINUE" | "FRAUD_REVIEW"
    threshold_version: str
    model_version: str
    inference_ms: float
    warnings: list[str] = field(default_factory=list)


def _load_thresholds(threshold_path: str | Path) -> dict:
    """Load and validate a threshold JSON file.

    Expected keys: version, low_threshold, high_threshold.

    Args:
        threshold_path: Path to thresholds_v1.json.

    Returns:
        Parsed threshold dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If required keys are missing.
    """
    path = Path(threshold_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Threshold file not found: {path}. "
            "Run notebook 03_fraud_evaluation_and_threshold.ipynb to generate it."
        )
    with open(path) as f:
        data = json.load(f)

    required = {"version", "low_threshold", "high_threshold"}
    missing = required - set(data.keys())
    if missing:
        raise KeyError(
            f"Threshold file {path} is missing required keys: {missing}. "
            f"Found keys: {list(data.keys())}"
        )
    return data


def _classify_risk(
    probability: float,
    low_threshold: float,
    high_threshold: float,
) -> tuple[str, str]:
    """Map a fraud probability to a risk level and routing decision.

    Args:
        probability: Sigmoid output [0, 1].
        low_threshold: Below this -> "low" risk.
        high_threshold: At or above this -> "high" risk -> FRAUD_REVIEW.

    Returns:
        Tuple of (risk_level, route).
    """
    if probability >= high_threshold:
        return "high", "FRAUD_REVIEW"
    elif probability >= low_threshold:
        return "medium", "CONTINUE"
    else:
        return "low", "CONTINUE"


def predict_fraud(
    image_path: str | Path,
    model_path: str | Path,
    threshold_path: str | Path,
    device: str = "cpu",
) -> FraudResult:
    """Run visual fraud-risk inference on a single image.

    Loads the model and thresholds, preprocesses the image with the
    standard val/test transform pipeline, and returns a FraudResult.

    Args:
        image_path: Path to the image file (JPEG, PNG, or WEBP).
        model_path: Path to the .pt checkpoint saved by notebook 02.
        threshold_path: Path to thresholds_v1.json saved by notebook 03.
        device: Torch device string ("cpu" or "cuda").

    Returns:
        FraudResult dataclass with probability, risk level, route, and metadata.

    Raises:
        FileNotFoundError: If image_path, model_path, or threshold_path is missing.
        RuntimeError: If the image cannot be decoded.
    """
    image_path = Path(image_path)
    warnings: list[str] = []

    # -- Validate image path --
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # -- Load thresholds --
    thresholds = _load_thresholds(threshold_path)
    low_thr = float(thresholds["low_threshold"])
    high_thr = float(thresholds["high_threshold"])
    threshold_version = str(thresholds.get("version", "unknown"))
    model_version = str(thresholds.get("model_version", "unknown"))

    # -- Load model --
    model = load_fraud_model(model_path, device=device)
    transform = get_transforms("val")  # deterministic for inference

    # -- Load and preprocess image --
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Failed to decode image {image_path}: {exc}") from exc

    w, h = image.size
    if w < 224 or h < 224:
        warnings.append(
            f"Image resolution ({w}x{h}) is below 224x224; "
            "transform will upscale which may reduce quality."
        )

    image_tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # -- Inference --
    start = time.perf_counter()
    with torch.no_grad():
        logit = model(image_tensor)  # (1, 1)
        probability = float(torch.sigmoid(logit).item())
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # -- Classify --
    risk_level, route = _classify_risk(probability, low_thr, high_thr)

    return FraudResult(
        image_path=str(image_path),
        fraud_probability=round(probability, 6),
        risk_level=risk_level,
        route=route,
        threshold_version=threshold_version,
        model_version=model_version,
        inference_ms=round(elapsed_ms, 2),
        warnings=warnings,
    )
