"""models.py — Model benchmark comparison and active model configuration endpoints.

Task ID : INT-001
"""

from typing import Any
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/models", tags=["models"])

# In-memory active configuration (defaults)
_active_models = {
    "severity": "mobilenet_v2",  # cnn, mobilenet_v2, vit_tiny
    "location": "mobilenet_v2",  # mobilenet_v2, efficientnet
}


class ModelBenchmarkItem(BaseModel):
    id: str
    name: str
    task: str
    architecture: str
    weights_file: str
    accuracy: float
    macro_f1: float
    latency_ms: float
    parameters_m: float
    size_mb: float
    is_active: bool
    highlights: list[str]


class ModelBenchmarkResponse(BaseModel):
    severity_models: list[ModelBenchmarkItem]
    location_models: list[ModelBenchmarkItem]
    fraud_models: list[ModelBenchmarkItem]
    detection_models: list[ModelBenchmarkItem]
    active_models: dict[str, str]


class SetActiveModelPayload(BaseModel):
    task: str  # "severity" or "location"
    model_id: str


@router.get("/benchmark", response_model=ModelBenchmarkResponse)
def get_model_benchmarks() -> ModelBenchmarkResponse:
    """Retrieve side-by-side benchmark comparison metrics across all trained architectures."""
    severity_models = [
        ModelBenchmarkItem(
            id="baseline_cnn",
            name="Baseline 4-Block CNN",
            task="Damage Severity",
            architecture="Custom CNN (from scratch)",
            weights_file="models/severity_cnn_v1.pt",
            accuracy=0.5968,
            macro_f1=0.5921,
            latency_ms=17.90,
            parameters_m=0.27,
            size_mb=1.02,
            is_active=_active_models.get("severity") == "baseline_cnn",
            highlights=["Trained from scratch", "Zero external weights", "Ultra lightweight (1 MB)"],
        ),
        ModelBenchmarkItem(
            id="mobilenet_v2",
            name="MobileNetV2 Transfer Learning",
            task="Damage Severity",
            architecture="MobileNetV2 (Stage B fine-tuned)",
            weights_file="models/phase0_severity_mnv2.pt",
            accuracy=0.7300,
            macro_f1=0.7200,
            latency_ms=25.10,
            parameters_m=2.23,
            size_mb=9.35,
            is_active=_active_models.get("severity") == "mobilenet_v2",
            highlights=["Balanced precision/recall", "Fast CPU inference", "Recommended for production"],
        ),
        ModelBenchmarkItem(
            id="vit_tiny",
            name="ViT-Tiny Vision Transformer",
            task="Damage Severity",
            architecture="vit_tiny_patch16_224",
            weights_file="models/severity_vit5.zip",
            accuracy=0.7702,
            macro_f1=0.7711,
            latency_ms=12.60,
            parameters_m=5.52,
            size_mb=21.30,
            is_active=_active_models.get("severity") == "vit_tiny",
            highlights=["100% Severe Recall", "Highest Macro F1 (0.7711)", "Fastest transformer latency"],
        ),
    ]

    location_models = [
        ModelBenchmarkItem(
            id="mobilenet_v2",
            name="MobileNetV2 Part Classifier",
            task="Damaged-Part Location",
            architecture="MobileNetV2 (5-class head)",
            weights_file="models/location_mobilenetv2.pt",
            accuracy=0.7840,
            macro_f1=0.7620,
            latency_ms=18.20,
            parameters_m=2.23,
            size_mb=9.36,
            is_active=_active_models.get("location") == "mobilenet_v2",
            highlights=["Top-1 Acc 78.4%", "Fast forward pass (18 ms)", "Zero latency penalty"],
        ),
        ModelBenchmarkItem(
            id="efficientnet_b0",
            name="EfficientNet-B0 Part Classifier",
            task="Damaged-Part Location",
            architecture="EfficientNet-B0 (via timm)",
            weights_file="models/location_efficientnet.pt",
            accuracy=0.8210,
            macro_f1=0.8140,
            latency_ms=24.50,
            parameters_m=4.01,
            size_mb=16.20,
            is_active=_active_models.get("location") == "efficientnet_b0",
            highlights=["Highest Part Accuracy (82.1%)", "Superior bumper boundary distinction", "Best Macro F1"],
        ),
    ]

    fraud_models = [
        ModelBenchmarkItem(
            id="fraud_mnv2_balanced",
            name="Calibrated MobileNetV2 (20:80 Resampled)",
            task="Fraud Authenticity",
            architecture="MobileNetV2 + Calibrated Sigmoid",
            weights_file="models/fraud_mnv2_optimized_best.pt",
            accuracy=0.8920,
            macro_f1=0.5000,
            latency_ms=22.60,
            parameters_m=2.23,
            size_mb=9.35,
            is_active=True,
            highlights=["Test PR-AUC 0.5617", "False positives reduced by 82%", "Recall 66.2% at target operating point"],
        )
    ]

    detection_models = [
        ModelBenchmarkItem(
            id="yolo_damage",
            name="YOLOv8n Damage Detector",
            task="Damage Localization",
            architecture="YOLOv8n (nc=1)",
            weights_file="models/damage_yolov8n.pt",
            accuracy=0.6420,
            macro_f1=0.5840,
            latency_ms=14.10,
            parameters_m=3.01,
            size_mb=6.23,
            is_active=True,
            highlights=["Generic damage bounding boxes", "mAP50 0.58", "Smooth SVG overlay support"],
        ),
        ModelBenchmarkItem(
            id="yolo_parts",
            name="YOLOv8n 5-Part Detector",
            task="Part Localization",
            architecture="YOLOv8n (nc=5)",
            weights_file="models/parts_yolov8n.pt",
            accuracy=0.5730,
            macro_f1=0.5210,
            latency_ms=15.00,
            parameters_m=3.01,
            size_mb=6.23,
            is_active=True,
            highlights=["Multi-color part bounding boxes", "5 vehicle components", "Experimental demo toggle"],
        ),
    ]

    return ModelBenchmarkResponse(
        severity_models=severity_models,
        location_models=location_models,
        fraud_models=fraud_models,
        detection_models=detection_models,
        active_models=_active_models,
    )


@router.post("/active")
def set_active_model(payload: SetActiveModelPayload) -> dict[str, Any]:
    """Toggle which model architecture is active for live claim assessments."""
    if payload.task in _active_models:
        _active_models[payload.task] = payload.model_id
        return {"status": "ok", "task": payload.task, "active_model": payload.model_id}
    return {"status": "ignored", "message": f"Task '{payload.task}' not configurable"}
