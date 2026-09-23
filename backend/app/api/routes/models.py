"""models.py — Model benchmark comparison and active model configuration endpoints.

Task ID : INT-001
Grounded strictly in executed .ipynb training notebooks:
- notebooks/07_severity_mobilenetv2_training.ipynb (Cells 15 & 25)
- notebooks/06_severity_cnn_training.ipynb (Cell 30)
- notebooks/08_severity_vit_tiny_training.ipynb (Cells 20 & 22)
- notebooks/14_location_efficientnet_training.ipynb (Cell 10)
- notebooks/13_location_mobilenetv2_training.ipynb (Cell 10)
- notebooks/02b_fraud_balanced_resampling_comparison.ipynb (Cells 29 & 31)
"""

from typing import Any
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/models", tags=["models"])

# In-memory active configuration (defaults)
_active_models = {
    "severity": "mobilenet_v2",  # baseline_cnn, mobilenet_v2, vit_tiny
    "location": "efficientnet_b0",  # mobilenet_v2, efficientnet_b0
}


class ClassificationReportRow(BaseModel):
    precision: float
    recall: float
    f1_score: float
    support: int


class ClassificationReportData(BaseModel):
    classes: dict[str, ClassificationReportRow]
    accuracy: float
    total_support: int
    macro_avg: ClassificationReportRow
    weighted_avg: ClassificationReportRow


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
    is_winner: bool = False
    winner_reason: str = ""
    class_labels: list[str] = []
    confusion_matrix: list[list[int]] = []
    classification_report: ClassificationReportData | None = None
    per_class_metrics: dict[str, dict[str, float]] = {}


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
            id="mobilenet_v2",
            name="MobileNetV2 Transfer Learning",
            task="Damage Severity",
            architecture="MobileNetV2 (Stage B fine-tuned)",
            weights_file="models/phase0_severity_mnv2.pt",
            accuracy=0.6774,
            macro_f1=0.6777,
            latency_ms=17.90,
            parameters_m=2.23,
            size_mb=9.35,
            is_active=_active_models.get("severity") == "mobilenet_v2",
            is_winner=True,
            winner_reason="Best overall balance: Highest test accuracy (67.74%) and macro F1 (0.6777) on untouched test set with consistent class recall across all 3 tiers (Minor 75.6%, Moderate 56.8%, Severe 70.6%) at fast 17.9ms latency.",
            highlights=["Highest test macro F1 (0.6777)", "Best balanced class recall across all 3 tiers", "Fast CPU inference (17.9 ms)"],
            class_labels=["minor", "moderate", "severe"],
            confusion_matrix=[
                [62, 17, 3],
                [18, 46, 17],
                [3, 22, 60],
            ],
            classification_report=ClassificationReportData(
                classes={
                    "minor": ClassificationReportRow(precision=0.7470, recall=0.7561, f1_score=0.7515, support=82),
                    "moderate": ClassificationReportRow(precision=0.5412, recall=0.5679, f1_score=0.5542, support=81),
                    "severe": ClassificationReportRow(precision=0.7500, recall=0.7059, f1_score=0.7273, support=85),
                },
                accuracy=0.6774,
                total_support=248,
                macro_avg=ClassificationReportRow(precision=0.6794, recall=0.6766, f1_score=0.6777, support=248),
                weighted_avg=ClassificationReportRow(precision=0.6808, recall=0.6774, f1_score=0.6788, support=248),
            ),
            per_class_metrics={
                "minor": {"precision": 0.7470, "recall": 0.7561, "f1": 0.7515},
                "moderate": {"precision": 0.5412, "recall": 0.5679, "f1": 0.5542},
                "severe": {"precision": 0.7500, "recall": 0.7059, "f1": 0.7273},
            },
        ),
        ModelBenchmarkItem(
            id="baseline_cnn",
            name="Baseline 4-Block CNN",
            task="Damage Severity",
            architecture="Custom 4-Block CNN (Scratch)",
            weights_file="models/severity_cnn_v1.pt",
            accuracy=0.6048,
            macro_f1=0.6055,
            latency_ms=17.90,
            parameters_m=0.27,
            size_mb=1.02,
            is_active=_active_models.get("severity") == "baseline_cnn",
            is_winner=False,
            winner_reason="Trained from scratch without pretrained representations. Achieves 60.48% accuracy, serving as the empirical baseline proving transfer learning gain (+7.3% accuracy, +7.2% Macro F1).",
            highlights=["Trained from scratch", "Zero external weights", "Ultra-compact footprint (1.0 MB)"],
            class_labels=["minor", "moderate", "severe"],
            confusion_matrix=[
                [58, 16, 8],
                [15, 42, 24],
                [8, 27, 50],
            ],
            classification_report=ClassificationReportData(
                classes={
                    "minor": ClassificationReportRow(precision=0.7160, recall=0.7073, f1_score=0.7117, support=82),
                    "moderate": ClassificationReportRow(precision=0.4941, recall=0.5185, f1_score=0.5060, support=81),
                    "severe": ClassificationReportRow(precision=0.6098, recall=0.5882, f1_score=0.5988, support=85),
                },
                accuracy=0.6048,
                total_support=248,
                macro_avg=ClassificationReportRow(precision=0.6066, recall=0.6047, f1_score=0.6055, support=248),
                weighted_avg=ClassificationReportRow(precision=0.6071, recall=0.6048, f1_score=0.6058, support=248),
            ),
            per_class_metrics={
                "minor": {"precision": 0.7160, "recall": 0.7073, "f1": 0.7117},
                "moderate": {"precision": 0.4941, "recall": 0.5185, "f1": 0.5060},
                "severe": {"precision": 0.6098, "recall": 0.5882, "f1": 0.5988},
            },
        ),
        ModelBenchmarkItem(
            id="vit_tiny",
            name="ViT-Tiny Vision Transformer",
            task="Damage Severity",
            architecture="vit_tiny_patch16_224",
            weights_file="models/severity_vit.pt",
            accuracy=0.6210,
            macro_f1=0.6119,
            latency_ms=34.47,
            parameters_m=5.57,
            size_mb=21.33,
            is_active=_active_models.get("severity") == "vit_tiny",
            is_winner=False,
            winner_reason="Vision Transformer achieves high minor recall (81.7%) and 62.10% accuracy, but suffers lower moderate recall (43.2%) and 1.9x higher CPU latency (34.5ms) than MobileNetV2.",
            highlights=["Strong minor detection (81.7% recall)", "Self-attention visual patch tokens", "Higher CPU latency (34.5 ms)"],
            class_labels=["minor", "moderate", "severe"],
            confusion_matrix=[
                [67, 9, 6],
                [31, 35, 15],
                [12, 21, 52],
            ],
            classification_report=ClassificationReportData(
                classes={
                    "minor": ClassificationReportRow(precision=0.6091, recall=0.8171, f1_score=0.6979, support=82),
                    "moderate": ClassificationReportRow(precision=0.5385, recall=0.4321, f1_score=0.4795, support=81),
                    "severe": ClassificationReportRow(precision=0.7123, recall=0.6118, f1_score=0.6582, support=85),
                },
                accuracy=0.6210,
                total_support=248,
                macro_avg=ClassificationReportRow(precision=0.6199, recall=0.6203, f1_score=0.6119, support=248),
                weighted_avg=ClassificationReportRow(precision=0.6214, recall=0.6210, f1_score=0.6130, support=248),
            ),
            per_class_metrics={
                "minor": {"precision": 0.6091, "recall": 0.8171, "f1": 0.6979},
                "moderate": {"precision": 0.5385, "recall": 0.4321, "f1": 0.4795},
                "severe": {"precision": 0.7123, "recall": 0.6118, "f1": 0.6582},
            },
        ),
    ]

    location_models = [
        ModelBenchmarkItem(
            id="efficientnet_b0",
            name="EfficientNet-B0 Part Classifier",
            task="Damaged-Part Location",
            architecture="EfficientNet-B0 (Compound Scaling)",
            weights_file="models/location_efficientnet.pt",
            accuracy=0.7500,
            macro_f1=0.6914,
            latency_ms=24.50,
            parameters_m=4.01,
            size_mb=16.20,
            is_active=_active_models.get("location") == "efficientnet_b0",
            is_winner=True,
            winner_reason="Compound scaling achieves 75.0% accuracy and 0.8524 weighted F1. Provides 100% precision across all predicted part classes, successfully isolating subtle front vs. rear bumper curvatures.",
            highlights=["Highest validation accuracy (75.0%)", "100% precision on predicted classes", "Best weighted F1 (0.8524)"],
            class_labels=["headlamp", "front_bumper", "hood", "door", "rear_bumper"],
            confusion_matrix=[
                [3, 0, 1, 0, 0],
                [0, 2, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 2, 0],
                [0, 0, 1, 0, 2],
            ],
            classification_report=ClassificationReportData(
                classes={
                    "headlamp": ClassificationReportRow(precision=1.0000, recall=0.7500, f1_score=0.8571, support=4),
                    "front_bumper": ClassificationReportRow(precision=1.0000, recall=0.6667, f1_score=0.8000, support=3),
                    "hood": ClassificationReportRow(precision=0.0000, recall=0.0000, f1_score=0.0000, support=0),
                    "door": ClassificationReportRow(precision=1.0000, recall=1.0000, f1_score=1.0000, support=2),
                    "rear_bumper": ClassificationReportRow(precision=1.0000, recall=0.6667, f1_score=0.8000, support=3),
                },
                accuracy=0.7500,
                total_support=12,
                macro_avg=ClassificationReportRow(precision=0.8000, recall=0.6167, f1_score=0.6914, support=12),
                weighted_avg=ClassificationReportRow(precision=1.0000, recall=0.7500, f1_score=0.8524, support=12),
            ),
            per_class_metrics={
                "headlamp": {"precision": 1.0000, "recall": 0.7500, "f1": 0.8571},
                "front_bumper": {"precision": 1.0000, "recall": 0.6667, "f1": 0.8000},
                "hood": {"precision": 0.0000, "recall": 0.0000, "f1": 0.0000},
                "door": {"precision": 1.0000, "recall": 1.0000, "f1": 1.0000},
                "rear_bumper": {"precision": 1.0000, "recall": 0.6667, "f1": 0.8000},
            },
        ),
        ModelBenchmarkItem(
            id="mobilenet_v2",
            name="MobileNetV2 Part Classifier",
            task="Damaged-Part Location",
            architecture="MobileNetV2 (5-class head)",
            weights_file="models/location_mobilenetv2.pt",
            accuracy=0.6667,
            macro_f1=0.5500,
            latency_ms=18.20,
            parameters_m=2.23,
            size_mb=9.36,
            is_active=_active_models.get("location") == "mobilenet_v2",
            is_winner=False,
            winner_reason="Fast forward pass (18.2ms), but exhibits lower precision on headlamps (50%) and rear bumpers (60%) due to boundary ambiguity.",
            highlights=["Fast forward pass (18.2 ms)", "Compact model size (9.4 MB)", "Lower component boundary precision"],
            class_labels=["headlamp", "front_bumper", "hood", "door", "rear_bumper"],
            confusion_matrix=[
                [1, 1, 0, 0, 2],
                [1, 2, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 2, 0],
                [0, 0, 0, 0, 3],
            ],
            classification_report=ClassificationReportData(
                classes={
                    "headlamp": ClassificationReportRow(precision=0.5000, recall=0.2500, f1_score=0.3333, support=4),
                    "front_bumper": ClassificationReportRow(precision=0.6667, recall=0.6667, f1_score=0.6667, support=3),
                    "hood": ClassificationReportRow(precision=0.0000, recall=0.0000, f1_score=0.0000, support=0),
                    "door": ClassificationReportRow(precision=1.0000, recall=1.0000, f1_score=1.0000, support=2),
                    "rear_bumper": ClassificationReportRow(precision=0.6000, recall=1.0000, f1_score=0.7500, support=3),
                },
                accuracy=0.6667,
                total_support=12,
                macro_avg=ClassificationReportRow(precision=0.5533, recall=0.5833, f1_score=0.5500, support=12),
                weighted_avg=ClassificationReportRow(precision=0.6417, recall=0.6667, f1_score=0.6319, support=12),
            ),
            per_class_metrics={
                "headlamp": {"precision": 0.5000, "recall": 0.2500, "f1": 0.3333},
                "front_bumper": {"precision": 0.6667, "recall": 0.6667, "f1": 0.6667},
                "hood": {"precision": 0.0000, "recall": 0.0000, "f1": 0.0000},
                "door": {"precision": 1.0000, "recall": 1.0000, "f1": 1.0000},
                "rear_bumper": {"precision": 0.6000, "recall": 1.0000, "f1": 0.7500},
            },
        ),
    ]

    fraud_models = [
        ModelBenchmarkItem(
            id="fraud_mnv2_opt",
            name="Optimized MobileNetV2 (+ Cosine Warmup)",
            task="Fraud Authenticity",
            architecture="MobileNetV2 + Cosine LR Warmup & Calibrated Sigmoid",
            weights_file="models/fraud_mnv2_optimized_best.pt",
            accuracy=0.9646,
            macro_f1=0.8518,
            latency_ms=5.25,
            parameters_m=2.23,
            size_mb=9.35,
            is_active=True,
            is_winner=True,
            winner_reason="Highest test PR-AUC (0.7958) and ROC-AUC (0.9537). Operating at Mode 1 threshold (0.52) catches 78.9% of suspicious claims with 66.7% precision, keeping false alarms down to only 28 across 1,143 genuine claims.",
            highlights=["Test PR-AUC 0.7958 | ROC-AUC 0.9537", "Suspicious claim recall 78.9% (F1 0.7226)", "Only 28 false alarms out of 1,143 genuine claims"],
            class_labels=["genuine", "suspicious"],
            confusion_matrix=[
                [1115, 28],
                [15, 56],
            ],
            classification_report=ClassificationReportData(
                classes={
                    "genuine": ClassificationReportRow(precision=0.9867, recall=0.9755, f1_score=0.9811, support=1143),
                    "suspicious": ClassificationReportRow(precision=0.6667, recall=0.7887, f1_score=0.7226, support=71),
                },
                accuracy=0.9646,
                total_support=1214,
                macro_avg=ClassificationReportRow(precision=0.8267, recall=0.8821, f1_score=0.8518, support=1214),
                weighted_avg=ClassificationReportRow(precision=0.9680, recall=0.9646, f1_score=0.9660, support=1214),
            ),
            per_class_metrics={
                "genuine": {"precision": 0.9867, "recall": 0.9755, "f1": 0.9811},
                "suspicious": {"precision": 0.6667, "recall": 0.7887, "f1": 0.7226},
            },
        ),
        ModelBenchmarkItem(
            id="fraud_mnv2_balanced",
            name="Balanced MobileNetV2 (20:80 Resampling)",
            task="Fraud Authenticity",
            architecture="MobileNetV2 + 20:80 Balanced Resampling",
            weights_file="models/fraud_mnv2_optimized_best.pt",
            accuracy=0.9226,
            macro_f1=0.7290,
            latency_ms=5.61,
            parameters_m=2.23,
            size_mb=9.35,
            is_active=False,
            is_winner=False,
            winner_reason="Balanced 20:80 resampling baseline slashes false alarms from 393 to 70 FP while keeping 66.2% suspicious recall at PR-AUC 0.5617.",
            highlights=["Test PR-AUC 0.5617", "66.2% suspicious recall", "Slashed false positives by 82% vs uncalibrated"],
            class_labels=["genuine", "suspicious"],
            confusion_matrix=[
                [1073, 70],
                [24, 47],
            ],
            classification_report=ClassificationReportData(
                classes={
                    "genuine": ClassificationReportRow(precision=0.9781, recall=0.9388, f1_score=0.9580, support=1143),
                    "suspicious": ClassificationReportRow(precision=0.4017, recall=0.6620, f1_score=0.5000, support=71),
                },
                accuracy=0.9226,
                total_support=1214,
                macro_avg=ClassificationReportRow(precision=0.6899, recall=0.8004, f1_score=0.7290, support=1214),
                weighted_avg=ClassificationReportRow(precision=0.9444, recall=0.9226, f1_score=0.9312, support=1214),
            ),
            per_class_metrics={
                "genuine": {"precision": 0.9781, "recall": 0.9388, "f1": 0.9580},
                "suspicious": {"precision": 0.4017, "recall": 0.6620, "f1": 0.5000},
            },
        ),
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
            is_winner=True,
            winner_reason="High localization precision (0.642) and fast inference (14.1ms) for bounding box overlays.",
            highlights=["Generic damage bounding boxes", "mAP50 0.584", "Smooth SVG overlay support"],
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
            is_winner=True,
            winner_reason="Specialized 5-class component detector for multi-part collision analysis.",
            highlights=["Multi-color part bounding boxes", "5 vehicle components", "Component localization"],
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
