"""quality — OpenCV-based image-quality and evidence-integrity checks.

Deterministic quality checks, blur estimation, brightness/contrast calculation,
orientation correction, duplicate identification, bounding-box drawing, and
the primary runtime orchestrator.

Phase 1 exports (audit-time):
    read_image_safely, calculate_blur_score, calculate_brightness,
    calculate_contrast, compute_image_metrics

Phase 3 exports (runtime):
    check_minimum_resolution, correct_orientation, extract_exif_summary,
    draw_bounding_boxes, save_annotated_image,
    QualityResult, run_quality_checks
"""

from claimvision_ml.quality.image_checks import (
    calculate_blur_score,
    calculate_brightness,
    calculate_contrast,
    check_minimum_resolution,
    compute_image_metrics,
    correct_orientation,
    draw_bounding_boxes,
    extract_exif_summary,
    read_image_safely,
    save_annotated_image,
)
from claimvision_ml.quality.runtime_checker import QualityResult, run_quality_checks

__all__ = [
    # Phase 1 — audit-time
    "read_image_safely",
    "calculate_blur_score",
    "calculate_brightness",
    "calculate_contrast",
    "compute_image_metrics",
    # Phase 3 — runtime checks (image_checks additions)
    "check_minimum_resolution",
    "correct_orientation",
    "extract_exif_summary",
    "draw_bounding_boxes",
    "save_annotated_image",
    # Phase 3 — runtime orchestrator
    "QualityResult",
    "run_quality_checks",
]
