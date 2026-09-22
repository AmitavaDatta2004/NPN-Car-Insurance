"""decision.py — Configurable decision routing engine for claim triage.

Task ID : INF-001
Phase   : 12 (Unified Inference)
Owner   : Member 5 (Backend/ML)

Implements the routing matrix specified in README §15 and config/decision_thresholds.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "fraud": {"low_threshold": 0.30, "high_threshold": 0.65},
    "severity": {"confidence_min": 0.40},
    "location": {"confidence_min": 0.40},
    "routing": {"fast_track_max_cost": 50000.0, "require_passed_quality": True},
}


def load_decision_thresholds(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configurable decision thresholds from YAML file."""
    if config_path is not None:
        p = Path(config_path)
    else:
        p = Path(__file__).resolve().parents[4] / "config" / "decision_thresholds.yaml"

    if p.is_file():
        try:
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return DEFAULT_THRESHOLDS


def route_claim(
    quality_acceptable: bool = True,
    quality_route: str = "CONTINUE",
    fraud_prob: float | None = None,
    severity_class: str | None = None,
    severity_conf: float | None = None,
    location_conf: float | None = None,
    cost_max: float | None = None,
    thresholds: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """Determine the triage route and reason codes for an assessed claim.

    Returns
    -------
    tuple[str, list[str]]
        (route, reason_codes)
    """
    if thresholds is None:
        thresholds = load_decision_thresholds()

    fraud_cfg = thresholds.get("fraud", DEFAULT_THRESHOLDS["fraud"])
    sev_cfg = thresholds.get("severity", DEFAULT_THRESHOLDS["severity"])
    loc_cfg = thresholds.get("location", DEFAULT_THRESHOLDS["location"])
    routing_cfg = thresholds.get("routing", DEFAULT_THRESHOLDS["routing"])

    fraud_high_thresh = float(fraud_cfg.get("high_threshold", 0.65))
    sev_conf_min = float(sev_cfg.get("confidence_min", 0.40))
    loc_conf_min = float(loc_cfg.get("confidence_min", 0.40))
    fast_track_max_cost = float(routing_cfg.get("fast_track_max_cost", 50000.0))

    # 1. Quality checks
    if not quality_acceptable or quality_route == "MORE_EVIDENCE_REQUIRED":
        return "MORE_EVIDENCE_REQUIRED", ["insufficient_evidence_quality"]

    if quality_route == "DUPLICATE_REVIEW":
        return "FRAUD_REVIEW", ["duplicate_image_detected"]

    # 2. Fraud check
    if fraud_prob is not None and fraud_prob >= fraud_high_thresh:
        return "FRAUD_REVIEW", ["high_fraud_risk"]

    # 3. Damage & Cost review checks
    manual_reasons: list[str] = []

    if severity_class is not None and severity_class.strip().lower() == "severe":
        manual_reasons.append("severe_damage")

    if cost_max is not None and cost_max > fast_track_max_cost:
        manual_reasons.append("cost_exceeds_threshold")

    if severity_conf is not None and severity_conf < sev_conf_min:
        manual_reasons.append("low_severity_confidence")

    if location_conf is not None and location_conf < loc_conf_min:
        manual_reasons.append("low_location_confidence")

    if manual_reasons:
        return "MANUAL_DAMAGE_REVIEW", manual_reasons

    # 4. Fast-track eligible
    reasons = ["acceptable_evidence_quality"]
    if fraud_prob is not None:
        reasons.append("low_fraud_risk")
    if severity_class is not None:
        reasons.append("damage_within_limits")
    if cost_max is not None:
        reasons.append("cost_within_limits")

    return "FAST_TRACK_ELIGIBLE", reasons
