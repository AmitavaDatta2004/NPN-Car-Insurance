"""pipeline — unified inference orchestrator.

Task ID : INF-001
Phase   : 12 (Unified Inference)

Public API:
    assess_claim        — run end-to-end claim assessment on an image
    route_claim         — decision routing engine
    AssessmentResult    — top-level structured assessment dataclass
    QualitySummary      — quality check summary
    FraudSummary        — fraud risk summary
    SeveritySummary     — severity classification summary
    LocationSummary     — location classification summary
    DetectionSummary    — damage localization summary
    CostSummary         — repair cost summary
"""

from claimvision_ml.pipeline.assess import assess_claim
from claimvision_ml.pipeline.decision import load_decision_thresholds, route_claim
from claimvision_ml.pipeline.schemas import (
    AssessmentResult,
    CostSummary,
    DetectionSummary,
    FraudSummary,
    LocationSummary,
    QualitySummary,
    SeveritySummary,
)

__all__ = [
    "assess_claim",
    "route_claim",
    "load_decision_thresholds",
    "AssessmentResult",
    "QualitySummary",
    "FraudSummary",
    "SeveritySummary",
    "LocationSummary",
    "DetectionSummary",
    "CostSummary",
]
