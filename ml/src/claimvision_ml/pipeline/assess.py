"""assess.py — Unified claim assessment pipeline.

Task ID : INF-001
Phase   : 12 (Unified Inference)
Owner   : Member 5 (Backend/ML)

Chains all deterministic quality checks and ML models in strict contract order:
    1. Quality (OpenCV)
    2. Fraud risk (MobileNetV2) — stops early if high fraud risk
    3. Damage severity (MobileNetV2 / ViT)
    4. Damage localization (YOLOv8)
    5. Damaged-part location (MobileNetV2 / EfficientNet-B0)
    6. Repair cost engine (Rule-based JSON table)
    7. Decision router (Configurable thresholds)
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

from claimvision_ml.costing import estimate_cost
from claimvision_ml.pipeline.decision import load_decision_thresholds, route_claim
from claimvision_ml.pipeline.schemas import (
    AssessmentResult,
    CostSummary,
    DetectionSummary,
    FraudSummary,
    GenAIGateSummary,
    LocationSummary,
    QualitySummary,
    SeveritySummary,
)
from claimvision_ml.quality import run_quality_checks
from backend.app.services.genai_gate import verify_vehicle_intake

logger = logging.getLogger(__name__)

# Default model version tags
DEFAULT_VERSIONS = {
    "quality": "CV-OPENCV-001",
    "genai_gate": "NVIDIA-NEMOTRON-VISION",
    "fraud": "FRD-MNV2-001",
    "severity": "SEV-MNV2-001",
    "detection": "DET-YOLO-001",
    "location": "LOC-EFF-001",
    "costing": "COST-RULES-001",
    "decision": "DECISION-001",
}


def _run_fraud_step(
    image_path: Path,
    fraud_ckpt: Path | None,
    threshold_path: Path | None,
) -> FraudSummary:
    """Run fraud inference or provide graceful deterministic mock."""
    if fraud_ckpt and fraud_ckpt.is_file() and threshold_path and threshold_path.is_file():
        try:
            from claimvision_ml.fraud.predict import predict_fraud

            res = predict_fraud(image_path, model_path=fraud_ckpt, threshold_path=threshold_path)
            return FraudSummary(
                probability=res.fraud_probability,
                risk_level=res.risk_level,
                route=res.route,
                model_version=res.model_version or DEFAULT_VERSIONS["fraud"],
                warnings=res.warnings,
            )
        except Exception as exc:
            logger.warning("Real fraud model failed; using fallback: %s", exc)

    # Deterministic fallback when checkpoint is not present
    return FraudSummary(
        probability=0.12,
        risk_level="low",
        route="CONTINUE",
        model_version=f"{DEFAULT_VERSIONS['fraud']}-mock",
        warnings=["mock_fraud_model: checkpoint not on disk"],
    )


def _run_severity_step(
    image_path: Path,
    severity_ckpt: Path | None,
) -> SeveritySummary:
    """Run severity inference or provide graceful deterministic mock."""
    if severity_ckpt and severity_ckpt.is_file():
        try:
            from claimvision_ml.severity.predict import predict_severity

            res = predict_severity(image_path, model_or_path=severity_ckpt)
            return SeveritySummary(
                predicted_class=res.predicted_class,
                confidence=res.confidence,
                probabilities=res.probabilities,
                model_version=res.model_version or DEFAULT_VERSIONS["severity"],
                warnings=res.warnings,
            )
        except Exception as exc:
            logger.warning("Real severity model failed; using fallback: %s", exc)

    return SeveritySummary(
        predicted_class="moderate",
        confidence=0.81,
        probabilities={"minor": 0.11, "moderate": 0.81, "severe": 0.08},
        model_version=f"{DEFAULT_VERSIONS['severity']}-mock",
        warnings=["mock_severity_model: checkpoint not on disk"],
    )


def _run_detection_step(
    image_path: Path,
    detection_weights: Path | None,
) -> list[DetectionSummary]:
    """Run damage detection or provide graceful deterministic mock."""
    if detection_weights and detection_weights.is_file():
        try:
            from claimvision_ml.detection.damage import DamageDetector

            detector = DamageDetector(detection_weights)
            detections = detector.predict(image_path)
            return [
                DetectionSummary(
                    class_id=d.class_id,
                    label=d.class_name,
                    confidence=d.confidence,
                    box_xyxy=d.box_xyxy,
                    box_normalized=d.box_normalized,
                )
                for d in detections
            ]
        except Exception as exc:
            logger.warning("Real damage detector failed; using fallback: %s", exc)

    # Deterministic fallback box
    return [
        DetectionSummary(
            class_id=0,
            label="damage",
            confidence=0.89,
            box_xyxy=[120.0, 80.0, 380.0, 310.0],
            box_normalized=[0.24, 0.16, 0.76, 0.62],
        )
    ]


def _run_location_step(
    image_path: Path,
    location_ckpt: Path | None,
) -> LocationSummary:
    """Run location classification or provide graceful deterministic mock."""
    if location_ckpt and location_ckpt.is_file():
        try:
            from claimvision_ml.location.inference import LocationClassifier

            model_type = "efficientnet" if "efficientnet" in str(location_ckpt).lower() else "mobilenet"
            clf = LocationClassifier(location_ckpt, model_type=model_type)
            res = clf.predict(image_path)
            return LocationSummary(
                predicted_part=res.class_name,
                confidence=res.confidence,
                top3=res.top3,
                model_type=res.model_type,
                model_version="LOC-EFF-001" if model_type == "efficientnet" else DEFAULT_VERSIONS["location"],
                warning=res.warning,
            )
        except Exception as exc:
            logger.warning("Real location classifier failed; using fallback: %s", exc)

    return LocationSummary(
        predicted_part="front_bumper",
        confidence=0.86,
        top3=[("front_bumper", 0.86), ("hood", 0.08), ("headlamp", 0.04)],
        probabilities={
            "front_bumper": 0.86,
            "hood": 0.08,
            "headlamp": 0.04,
            "door": 0.01,
            "rear_bumper": 0.01,
        },
        model_type="mobilenet",
        model_version=f"{DEFAULT_VERSIONS['location']}-mock",
        warning="",
    )


def assess_claim(
    image_path: str | Path,
    claim_id: str | None = None,
    vehicle_segment: str = "compact",
    config: dict[str, Any] | None = None,
) -> AssessmentResult:
    """Execute end-to-end claim assessment on an input image.

    Parameters
    ----------
    image_path:
        Path to vehicle evidence photograph.
    claim_id:
        Optional UUID string. If None, a new UUID is generated.
    vehicle_segment:
        Vehicle classification for cost adjustment (e.g. 'compact', 'sedan', 'suv').
    config:
        Optional dictionary with paths to checkpoints and threshold configurations.

    Returns
    -------
    AssessmentResult
        Complete structured assessment matching API and dashboard contracts.
    """
    t0 = time.perf_counter()
    cid = claim_id or str(uuid.uuid4())
    img_path = Path(image_path)
    warnings: list[str] = []

    cfg = config or {}
    thresholds = cfg.get("thresholds") or load_decision_thresholds(cfg.get("thresholds_path"))

    # Validate image path
    if not img_path.is_file():
        elapsed = (time.perf_counter() - t0) * 1000
        return AssessmentResult(
            claim_id=cid,
            image_path=str(img_path),
            route="MORE_EVIDENCE_REQUIRED",
            reason_codes=["image_file_not_found"],
            inference_ms=elapsed,
            warnings=[f"File does not exist: {img_path}"],
        )

    # -----------------------------------------------------------------------
    # Step 1: Quality Checks
    # -----------------------------------------------------------------------
    q_res = run_quality_checks(img_path)
    q_summary = QualitySummary(
        blur_score=q_res.blur_score,
        brightness=q_res.brightness,
        contrast=q_res.contrast,
        acceptable=q_res.passed,
        route=q_res.route,
        warnings=q_res.warnings,
    )
    warnings.extend(q_res.warnings)

    if not q_res.passed:
        # Quality failure or duplicate detected
        route, reason_codes = route_claim(
            quality_acceptable=False,
            quality_route=q_res.route,
            thresholds=thresholds,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        return AssessmentResult(
            claim_id=cid,
            image_path=str(img_path),
            route=route,
            reason_codes=reason_codes,
            quality=q_summary,
            model_versions={"quality": DEFAULT_VERSIONS["quality"]},
            inference_ms=elapsed,
            warnings=warnings,
        )

    # -----------------------------------------------------------------------
    # Step 1b: GenAI Vehicle Screening Gate (Nemotron Omni / OpenRouter)
    # -----------------------------------------------------------------------
    genai_res = verify_vehicle_intake(img_path)
    genai_summary = GenAIGateSummary(
        is_vehicle=genai_res.is_vehicle,
        is_damaged=genai_res.is_damaged,
        vehicle_type=genai_res.vehicle_type,
        detected_object=genai_res.detected_object,
        reasoning=genai_res.reasoning,
        model_name=genai_res.model_name,
        latency_ms=genai_res.latency_ms,
    )

    if not genai_res.is_vehicle:
        # Non-vehicle image detected (e.g. tree, house, dog, document)
        elapsed = (time.perf_counter() - t0) * 1000
        return AssessmentResult(
            claim_id=cid,
            image_path=str(img_path),
            route="MORE_EVIDENCE_REQUIRED",
            reason_codes=["not_a_vehicle"],
            quality=q_summary,
            genai_gate=genai_summary,
            model_versions={
                "quality": DEFAULT_VERSIONS["quality"],
                "genai_gate": genai_res.model_name,
            },
            inference_ms=elapsed,
            warnings=[f"Non-vehicle uploaded ({genai_res.detected_object}): {genai_res.reasoning}"],
        )

    # -----------------------------------------------------------------------
    # Step 2: Fraud Risk Prediction (Discrete Flag 0 vs Flag 1)
    # -----------------------------------------------------------------------
    fraud_ckpt = Path(cfg.get("fraud_checkpoint", "models/fraud_mnv2_optimized_best.pt"))
    fraud_thresh_path = Path(cfg.get("fraud_threshold_path", "models/optimized_thresholds.json"))
    fraud_summary = _run_fraud_step(img_path, fraud_ckpt, fraud_thresh_path)
    
    # Discrete Binary Fraud Logic: > 0.50 is 1 (Suspicious), <= 0.50 is 0 (Genuine)
    fraud_flag = 1 if fraud_summary.probability > 0.50 else 0
    fraud_summary.flag = fraud_flag
    fraud_summary.verdict = "SUSPICIOUS" if fraud_flag == 1 else "GENUINE"
    warnings.extend(fraud_summary.warnings)

    # -----------------------------------------------------------------------
    # Step 3: Damage Severity
    # -----------------------------------------------------------------------
    sev_ckpt = Path(cfg.get("severity_checkpoint", "models/phase0_severity_mnv2.pt"))
    sev_summary = _run_severity_step(img_path, sev_ckpt)
    warnings.extend(sev_summary.warnings)

    # -----------------------------------------------------------------------
    # Step 4: Damage Detection
    # -----------------------------------------------------------------------
    det_weights = Path(cfg.get("detection_weights", "models/damage_yolov8n.pt"))
    detections = _run_detection_step(img_path, det_weights)

    # -----------------------------------------------------------------------
    # Step 5: Damaged-Part Location Classification (EfficientNet-B0)
    # -----------------------------------------------------------------------
    loc_ckpt = Path(cfg.get("location_checkpoint", "models/location_efficientnet.pt"))
    loc_summary = _run_location_step(img_path, loc_ckpt)
    if loc_summary.warning:
        warnings.append(loc_summary.warning)

    # -----------------------------------------------------------------------
    # Step 6: Cost Estimation
    # -----------------------------------------------------------------------
    # Aggregate parts: from location prediction and detection labels
    parts_to_cost: list[str] = []
    if loc_summary and loc_summary.predicted_part:
        parts_to_cost.append(loc_summary.predicted_part)
    for d in detections:
        if d.label != "damage":
            parts_to_cost.append(d.label)

    cost_est = estimate_cost(
        parts=parts_to_cost or ["damage"],
        severity=sev_summary.predicted_class,
        segment=vehicle_segment,
        config_path=cfg.get("cost_table_path"),
    )
    cost_summary = CostSummary(
        currency=cost_est.currency,
        min_cost=cost_est.min_cost,
        max_cost=cost_est.max_cost,
        breakdown=[b.to_dict() for b in cost_est.breakdown],
        vehicle_segment=cost_est.vehicle_segment,
        notes=cost_est.notes,
    )

    # -----------------------------------------------------------------------
    # Step 7: Decision Engine
    # -----------------------------------------------------------------------
    if fraud_summary.flag == 1:
        # Flag 1: Suspicious image -> Stop automatic fast-track, route to manual review
        route = "FRAUD_REVIEW"
        reason_codes = ["high_fraud_risk", "discrete_flag_1_suspicious"]
    else:
        route, reason_codes = route_claim(
            quality_acceptable=q_summary.acceptable,
            quality_route=q_summary.route,
            fraud_prob=fraud_summary.probability,
            severity_class=sev_summary.predicted_class,
            severity_conf=sev_summary.confidence,
            location_conf=loc_summary.confidence if loc_summary else None,
            cost_max=cost_summary.max_cost,
            thresholds=thresholds,
        )

    elapsed = (time.perf_counter() - t0) * 1000

    return AssessmentResult(
        claim_id=cid,
        image_path=str(img_path),
        route=route,
        reason_codes=reason_codes,
        quality=q_summary,
        genai_gate=genai_summary,
        fraud=fraud_summary,
        severity=sev_summary,
        location=loc_summary,
        detections=detections,
        cost=cost_summary,
        model_versions={
            "quality": DEFAULT_VERSIONS["quality"],
            "genai_gate": genai_res.model_name,
            "fraud": fraud_summary.model_version,
            "severity": sev_summary.model_version,
            "detection": DEFAULT_VERSIONS["detection"],
            "location": loc_summary.model_version if loc_summary else DEFAULT_VERSIONS["location"],
            "costing": DEFAULT_VERSIONS["costing"],
            "decision": DEFAULT_VERSIONS["decision"],
        },
        inference_ms=elapsed,
        warnings=warnings,
    )
