"""fraud - visual fraud-risk classifier and deterministic integrity signals.

Implemented in Phase 2 (notebooks 02 and 03).

Public API
----------
FraudClassifier     MobileNetV2 + custom head (nn.Module).
build_fraud_model   Build a new model (frozen backbone, Stage A ready).
load_fraud_model    Load from a .pt checkpoint.
export_onnx         Export to ONNX for backend inference.
FraudDataset        PyTorch Dataset reading frozen manifest CSVs.
get_class_weights   Compute inverse-frequency pos_weight for BCEWithLogitsLoss.
get_transforms      Augmentation pipeline for train / val / test.
FraudResult         Dataclass returned by predict_fraud.
predict_fraud       Primary runtime function called by the FastAPI backend.
"""

from claimvision_ml.fraud.dataset import FraudDataset, get_class_weights, get_transforms
from claimvision_ml.fraud.model import (
    FraudClassifier,
    build_fraud_model,
    export_onnx,
    load_fraud_model,
    save_preprocessing_config,
)
from claimvision_ml.fraud.predict import FraudResult, predict_fraud

__all__ = [
    "FraudClassifier",
    "build_fraud_model",
    "load_fraud_model",
    "export_onnx",
    "save_preprocessing_config",
    "FraudDataset",
    "get_class_weights",
    "get_transforms",
    "FraudResult",
    "predict_fraud",
]
