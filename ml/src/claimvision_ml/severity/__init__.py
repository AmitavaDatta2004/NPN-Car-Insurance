"""severity — damage severity classifiers: baseline CNN, MobileNetV2, ViT-Tiny.

Implemented in Phases 5–7 (notebooks 06–09).

Public API
----------
SeverityMobileNetV2        MobileNetV2 + custom 3-class head (nn.Module).
build_severity_mobilenet   Factory function to construct model.
load_severity_checkpoint   Load model from .pt checkpoint.
save_severity_checkpoint   Save model and metadata to .pt checkpoint.
export_onnx                Export model to ONNX with parity verification.
SeverityDataset            PyTorch Dataset reading frozen CSV manifests.
get_severity_transforms    Train / Val / Test augmentation pipelines.
get_severity_class_weights Compute balanced class weights for CrossEntropyLoss.
SEVERITY_CLASSES           Tuple of class names: ("minor", "moderate", "severe").
SEVERITY_CLASS_TO_ID       Dictionary mapping class name to integer ID.
SEVERITY_ID_TO_CLASS       Dictionary mapping integer ID to class name.
SeverityResult             Dataclass returned by predict_severity.
predict_severity           Standalone runtime function for backend & notebooks.
"""

from __future__ import annotations

from claimvision_ml.severity.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SEVERITY_CLASS_TO_ID,
    SEVERITY_CLASSES,
    SEVERITY_ID_TO_CLASS,
    SeverityDataset,
    get_severity_class_weights,
    get_severity_transforms,
)
from claimvision_ml.severity.mobilenet import (
    SeverityMobileNetV2,
    build_severity_mobilenet,
    export_onnx,
    load_severity_checkpoint,
    save_severity_checkpoint,
)
from claimvision_ml.severity.predict import SeverityResult, predict_severity

__all__ = [
    "SeverityMobileNetV2",
    "build_severity_mobilenet",
    "load_severity_checkpoint",
    "save_severity_checkpoint",
    "export_onnx",
    "SeverityDataset",
    "get_severity_transforms",
    "get_severity_class_weights",
    "SEVERITY_CLASSES",
    "SEVERITY_CLASS_TO_ID",
    "SEVERITY_ID_TO_CLASS",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "SeverityResult",
    "predict_severity",
]
