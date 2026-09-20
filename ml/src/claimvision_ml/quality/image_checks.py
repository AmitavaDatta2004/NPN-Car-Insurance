"""Deterministic OpenCV image quality and evidence checks.

Provides safe image loading, decodability verification, dimension extraction,
Laplacian blur scoring, brightness calculation, and contrast calculation.
Preserves original image data without silent modification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_image_safely(
    image_path: str | Path,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Attempt to decode an image with OpenCV safely.

    Args:
        image_path: Path to the image file.

    Returns:
        A tuple of (bgr_image, metadata_dict). If unreadable or corrupt,
        bgr_image is None and metadata_dict contains error information.
    """
    path_str = str(image_path)
    meta: dict[str, Any] = {
        "path": path_str,
        "is_valid": False,
        "width": 0,
        "height": 0,
        "channels": 0,
        "aspect_ratio": 0.0,
        "error": None,
    }

    p = Path(image_path)
    if not p.exists() or not p.is_file():
        meta["error"] = "file_not_found"
        return None, meta

    if p.stat().st_size == 0:
        meta["error"] = "empty_file"
        return None, meta

    try:
        # cv2.imread returns None on corrupt or unsupported images
        image = cv2.imread(path_str, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            meta["error"] = "corrupt_or_undecodable"
            return None, meta

        h, w = image.shape[:2]
        c = image.shape[2] if image.ndim == 3 else 1

        meta["is_valid"] = True
        meta["width"] = int(w)
        meta["height"] = int(h)
        meta["channels"] = int(c)
        meta["aspect_ratio"] = round(float(w) / float(h), 4) if h > 0 else 0.0
        return image, meta

    except Exception as exc:
        meta["error"] = f"decode_exception: {exc}"
        return None, meta


def calculate_blur_score(image: np.ndarray) -> float:
    """Calculate the blur score using variance of the Laplacian on grayscale.

    Higher variance corresponds to sharper edges; lower variance indicates blur.
    """
    if image is None or image.size == 0:
        return 0.0

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())
    return round(variance, 4)


def calculate_brightness(image: np.ndarray) -> float:
    """Calculate mean brightness of the image (0.0 to 255.0)."""
    if image is None or image.size == 0:
        return 0.0

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    return round(float(np.mean(gray)), 4)


def calculate_contrast(image: np.ndarray) -> float:
    """Calculate contrast as the standard deviation of grayscale intensity."""
    if image is None or image.size == 0:
        return 0.0

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    return round(float(np.std(gray)), 4)


def compute_image_metrics(image_path: str | Path) -> dict[str, Any]:
    """Inspect an image and return complete quality metrics dictionary.

    Returns:
        Dictionary with status, dimensions, blur_score, brightness, and contrast.
    """
    image, meta = read_image_safely(image_path)
    if not meta["is_valid"] or image is None:
        meta.update(
            {
                "blur_score": 0.0,
                "brightness": 0.0,
                "contrast": 0.0,
            }
        )
        return meta

    meta["blur_score"] = calculate_blur_score(image)
    meta["brightness"] = calculate_brightness(image)
    meta["contrast"] = calculate_contrast(image)
    return meta
