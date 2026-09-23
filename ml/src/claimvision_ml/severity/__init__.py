"""severity — damage severity classifiers: baseline CNN, MobileNetV2, ViT-Tiny.

Implemented in Phases 5–7 (notebooks 06–09).

Public API
----------
Baseline CNN (Phase 5):
  SeverityCNN, build_cnn_model, load_cnn_model, save_cnn_checkpoint,
  export_cnn_onnx, save_cnn_preprocessing_config, predict_severity_cnn,
  measure_cpu_latency, CNN_IMAGE_SIZE

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

from claimvision_ml.severity.cnn import (
    CNN_IMAGE_SIZE,
    # new improvement: native 4:3 aspect ratio matching dataset median
    CNN_IMAGE_SIZE_4_3,
    SEBlock,
    SeverityCNN,
    build_cnn_model,
    export_onnx as export_cnn_onnx,
    load_cnn_model,
    measure_cpu_latency,
    predict_severity_cnn,
    save_cnn_checkpoint,
    save_cnn_preprocessing_config,
)
from claimvision_ml.severity.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    LetterboxResize,
    SEVERITY_CLASS_TO_ID,
    SEVERITY_CLASSES,
    SEVERITY_ID_TO_CLASS,
    SeverityDataset,
    build_letterbox_transforms,
    create_severity_dataloader,
    get_severity_class_weights,
    get_severity_transforms,
    resolve_image_path,
    seed_worker,
)
from claimvision_ml.severity.mobilenet import (
    HybridOrdinalLoss,
    SeverityMobileNetV2,
    audit_batchnorm,
    build_severity_mobilenet,
    compute_ordinal_error_metrics,
    export_onnx,
    load_severity_checkpoint,
    make_class_weights,
    save_severity_checkpoint,
    set_frozen_batchnorm_eval,
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
    "seed_worker",
    "create_severity_dataloader",
    "LetterboxResize",
    "build_letterbox_transforms",
    # Baseline CNN (Phase 5)
    "CNN_IMAGE_SIZE",
    # new improvement: native 4:3 aspect ratio matching dataset median
    "CNN_IMAGE_SIZE_4_3",
    "SEBlock",
    "SeverityCNN",
    "build_cnn_model",
    "load_cnn_model",
    "save_cnn_checkpoint",
    "export_cnn_onnx",
    "save_cnn_preprocessing_config",
    "predict_severity_cnn",
    "measure_cpu_latency",
    # MobileNetV2 (Phase 6)
    "SeverityMobileNetV2",
    "audit_batchnorm",
    "build_severity_mobilenet",
    "load_severity_checkpoint",
    "save_severity_checkpoint",
    "set_frozen_batchnorm_eval",
    "make_class_weights",
    "HybridOrdinalLoss",
    "compute_ordinal_error_metrics",
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
