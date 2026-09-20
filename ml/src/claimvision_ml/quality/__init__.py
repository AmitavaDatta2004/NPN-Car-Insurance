"""quality — OpenCV-based image-quality and evidence-integrity checks.

Deterministic quality checks, blur estimation, brightness/contrast calculation,
and duplicate identification.
"""

from claimvision_ml.quality.image_checks import (
    calculate_blur_score,
    calculate_brightness,
    calculate_contrast,
    compute_image_metrics,
    read_image_safely,
)

__all__ = [
    "read_image_safely",
    "calculate_blur_score",
    "calculate_brightness",
    "calculate_contrast",
    "compute_image_metrics",
]
