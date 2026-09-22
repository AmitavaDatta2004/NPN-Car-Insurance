"""location/__init__.py — Public API for the ClaimVision AI location classifier.

Task IDs : LOC-DATA-001, LOC-MNV2-001, LOC-EFF-001
Phase    : 10b / 11a / 11b
Owner    : Detection ML member

This subpackage provides image-level classification of damaged vehicle parts.
Unlike the YOLO detection module (which draws bounding boxes), this module
classifies the whole image into one of five part classes.

Public API
----------
LOCATION_CLASSES      : list[str]  — 5 class names in label-id order
LocationClassification : dataclass  — inference result with class, confidence, top-3
LocationClassifier     : class      — runtime inference wrapper (MNV2 or EfficientNet)
LocationDataset        : class      — dataset loader with dominant-part label derivation
derive_location_labels : function   — derive single label per image from COCO JSON
classify_location      : function   — functional inference API
export_location_onnx   : function   — export checkpoint to ONNX
"""

from __future__ import annotations

from claimvision_ml.location.dataset import (
    LOCATION_CLASSES,
    LocationDataset,
    derive_location_labels,
    get_location_transforms,
    load_location_splits,
)
from claimvision_ml.location.inference import (
    LocationClassification,
    LocationClassifier,
    classify_location,
    export_location_onnx,
)

__all__ = [
    "LOCATION_CLASSES",
    "LocationDataset",
    "LocationClassification",
    "LocationClassifier",
    "derive_location_labels",
    "get_location_transforms",
    "load_location_splits",
    "classify_location",
    "export_location_onnx",
]
