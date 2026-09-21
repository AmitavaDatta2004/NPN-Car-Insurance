"""detection — YOLO-based generic damage localisation and damaged-part detection.

Implemented in Phases 8–10 (notebooks 10–12).

Public API:
    COCOtoYOLOConverter  — audit, convert, and verify COCO → YOLO (Phase 8)
    ValidationResult     — dataclass returned by validate_structure
    detect_damage(image_path, model_path) -> list[Detection]   (Phase 9)
    detect_parts(image_path, model_path)  -> list[Detection]   (Phase 10)
"""

from claimvision_ml.detection.coco_converter import COCOtoYOLOConverter, ValidationResult
from claimvision_ml.detection.damage import (
    DamageDetection,
    DamageDetector,
    detect_damage,
    export_damage_onnx,
    normalized_to_xyxy,
    xyxy_to_normalized,
)

__all__ = [
    "COCOtoYOLOConverter",
    "ValidationResult",
    "DamageDetection",
    "DamageDetector",
    "detect_damage",
    "export_damage_onnx",
    "xyxy_to_normalized",
    "normalized_to_xyxy",
]
