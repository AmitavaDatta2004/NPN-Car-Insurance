"""severity — damage severity classifiers: baseline CNN, MobileNetV2, ViT-Tiny.

Implemented in Phases 5–7 (notebooks 06–09).

Public API
----------
MobileNetV2 (Phase 6):
  SeverityMobileNetV2, build_severity_mobilenet, load_severity_checkpoint,
  save_severity_checkpoint, export_onnx, predict_severity, SeverityResult

ViT-Tiny (Phase 7):
  SeverityViTTiny, build_vit_model, load_vit_model, save_vit_checkpoint,
  export_vit_onnx, predict_severity_vit, SeverityViTResult, get_vit_transforms

Shared:
  SeverityDataset, SEVERITY_CLASSES, SEVERITY_CLASS_TO_ID, SEVERITY_ID_TO_CLASS
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
from claimvision_ml.severity.vit import (
    CLASS_TO_ID,
    ID_TO_CLASS,
    SeverityViTResult,
    SeverityViTTiny,
    build_vit_model,
    export_vit_onnx,
    get_vit_transforms,
    load_vit_model,
    predict_severity_vit,
    save_vit_checkpoint,
)

__all__ = [
    # Common taxonomy
    "SEVERITY_CLASSES",
    "SEVERITY_CLASS_TO_ID",
    "SEVERITY_ID_TO_CLASS",
    "CLASS_TO_ID",
    "ID_TO_CLASS",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "SeverityDataset",
    "get_severity_transforms",
    "get_severity_class_weights",
    # MobileNetV2 (Phase 6)
    "SeverityMobileNetV2",
    "build_severity_mobilenet",
    "load_severity_checkpoint",
    "save_severity_checkpoint",
    "export_onnx",
    "SeverityResult",
    "predict_severity",
    # ViT-Tiny (Phase 7)
    "SeverityViTTiny",
    "build_vit_model",
    "load_vit_model",
    "save_vit_checkpoint",
    "export_vit_onnx",
    "predict_severity_vit",
    "SeverityViTResult",
    "get_vit_transforms",
]
