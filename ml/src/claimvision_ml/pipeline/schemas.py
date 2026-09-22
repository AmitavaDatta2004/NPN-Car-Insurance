"""schemas.py — Standard data structures and serialization contracts for assessment.

Task ID : INF-001
Phase   : 12 (Unified Inference)
Owner   : Member 5 (Backend/ML)

Defines AssessmentResult and child dataclasses, matching the API contracts
required by backend routes and frontend components.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class QualitySummary:
    """Summary of OpenCV-based image quality checks."""

    blur_score: float
    brightness: float
    contrast: float
    acceptable: bool
    route: str = "CONTINUE"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "blur_score": round(self.blur_score, 2),
            "brightness": round(self.brightness, 2),
            "contrast": round(self.contrast, 2),
            "acceptable": self.acceptable,
            "route": self.route,
            "warnings": self.warnings,
        }


@dataclass
class FraudSummary:
    """Summary of fraud risk classifier."""

    probability: float
    risk_level: str  # "low" | "medium" | "high"
    route: str  # "CONTINUE" | "FRAUD_REVIEW"
    model_version: str = "FRD-MNV2-001"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability": round(self.probability, 4),
            "risk_level": self.risk_level,
            "route": self.route,
            "model_version": self.model_version,
            "warnings": self.warnings,
        }


@dataclass
class SeveritySummary:
    """Summary of damage severity classification."""

    predicted_class: str  # "minor" | "moderate" | "severe"
    confidence: float
    probabilities: dict[str, float]
    model_version: str = "SEV-MNV2-001"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "model_version": self.model_version,
            "warnings": self.warnings,
        }


@dataclass
class LocationSummary:
    """Summary of dominant damaged-part location classifier."""

    predicted_part: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    top3: list[tuple[str, float]] = field(default_factory=list)
    model_type: str = "mobilenet"
    model_version: str = "LOC-MNV2-001"
    warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_part": self.predicted_part,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "top3": [(part, round(p, 4)) for part, p in self.top3],
            "model_type": self.model_type,
            "model_version": self.model_version,
            "warning": self.warning,
        }


@dataclass
class DetectionSummary:
    """Summary of a single localized damage region."""

    class_id: int
    label: str
    confidence: float
    box_xyxy: list[float]
    box_normalized: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "box_xyxy": [round(v, 2) for v in self.box_xyxy],
            "box_normalized": [round(v, 4) for v in self.box_normalized],
        }


@dataclass
class CostSummary:
    """Summary of rule-based repair/replace cost estimation."""

    currency: str
    min_cost: float
    max_cost: float
    breakdown: list[dict[str, Any]] = field(default_factory=list)
    vehicle_segment: str = "compact"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "min_cost": round(self.min_cost, 2),
            "max_cost": round(self.max_cost, 2),
            "vehicle_segment": self.vehicle_segment,
            "breakdown": self.breakdown,
            "notes": self.notes,
        }


@dataclass
class AssessmentResult:
    """Unified assessment output produced by assess_claim()."""

    claim_id: str
    image_path: str
    route: str
    reason_codes: list[str]
    quality: QualitySummary | None = None
    fraud: FraudSummary | None = None
    severity: SeveritySummary | None = None
    location: LocationSummary | None = None
    detections: list[DetectionSummary] = field(default_factory=list)
    cost: CostSummary | None = None
    model_versions: dict[str, str] = field(default_factory=dict)
    inference_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert entire assessment to clean dictionary for JSON serialization."""
        return {
            "claim_id": self.claim_id,
            "image_path": self.image_path,
            "route": self.route,
            "reason_codes": self.reason_codes,
            "quality": self.quality.to_dict() if self.quality else None,
            "fraud": self.fraud.to_dict() if self.fraud else None,
            "severity": self.severity.to_dict() if self.severity else None,
            "location": self.location.to_dict() if self.location else None,
            "detections": [d.to_dict() for d in self.detections],
            "cost": self.cost.to_dict() if self.cost else None,
            "model_versions": self.model_versions,
            "inference_ms": round(self.inference_ms, 2),
            "warnings": self.warnings,
        }
