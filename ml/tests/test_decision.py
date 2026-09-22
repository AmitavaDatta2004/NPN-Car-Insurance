"""Unit tests for decision routing engine."""

from __future__ import annotations

import pytest

from claimvision_ml.pipeline.decision import load_decision_thresholds, route_claim


def test_load_decision_thresholds_defaults():
    """Verify decision thresholds load correctly."""
    thresholds = load_decision_thresholds()
    assert "fraud" in thresholds
    assert "severity" in thresholds
    assert "location" in thresholds
    assert "routing" in thresholds


def test_route_unacceptable_quality():
    """Verify unacceptable quality routes to MORE_EVIDENCE_REQUIRED."""
    route, reasons = route_claim(quality_acceptable=False)
    assert route == "MORE_EVIDENCE_REQUIRED"
    assert "insufficient_evidence_quality" in reasons


def test_route_duplicate_image():
    """Verify duplicate image routes to FRAUD_REVIEW."""
    route, reasons = route_claim(
        quality_acceptable=True,
        quality_route="DUPLICATE_REVIEW",
    )
    assert route == "FRAUD_REVIEW"
    assert "duplicate_image_detected" in reasons


def test_route_high_fraud_risk():
    """Verify high fraud probability routes to FRAUD_REVIEW."""
    route, reasons = route_claim(
        quality_acceptable=True,
        fraud_prob=0.85,
    )
    assert route == "FRAUD_REVIEW"
    assert "high_fraud_risk" in reasons


def test_route_severe_damage():
    """Verify severe damage routes to MANUAL_DAMAGE_REVIEW."""
    route, reasons = route_claim(
        quality_acceptable=True,
        fraud_prob=0.10,
        severity_class="severe",
        severity_conf=0.90,
        location_conf=0.85,
        cost_max=25000.0,
    )
    assert route == "MANUAL_DAMAGE_REVIEW"
    assert "severe_damage" in reasons


def test_route_cost_exceeds_threshold():
    """Verify excessive repair cost routes to MANUAL_DAMAGE_REVIEW."""
    route, reasons = route_claim(
        quality_acceptable=True,
        fraud_prob=0.10,
        severity_class="moderate",
        severity_conf=0.85,
        location_conf=0.80,
        cost_max=75000.0,  # exceeds default 50000.0 limit
    )
    assert route == "MANUAL_DAMAGE_REVIEW"
    assert "cost_exceeds_threshold" in reasons


def test_route_low_model_confidence():
    """Verify low severity or location confidence routes to MANUAL_DAMAGE_REVIEW."""
    route, reasons = route_claim(
        quality_acceptable=True,
        fraud_prob=0.10,
        severity_class="moderate",
        severity_conf=0.30,  # below default 0.40 min
        location_conf=0.80,
        cost_max=20000.0,
    )
    assert route == "MANUAL_DAMAGE_REVIEW"
    assert "low_severity_confidence" in reasons

    route2, reasons2 = route_claim(
        quality_acceptable=True,
        fraud_prob=0.10,
        severity_class="moderate",
        severity_conf=0.80,
        location_conf=0.25,  # below min
        cost_max=20000.0,
    )
    assert route2 == "MANUAL_DAMAGE_REVIEW"
    assert "low_location_confidence" in reasons2


def test_route_fast_track_eligible():
    """Verify clean, low-risk, moderate-cost claim qualifies for FAST_TRACK_ELIGIBLE."""
    route, reasons = route_claim(
        quality_acceptable=True,
        fraud_prob=0.12,
        severity_class="minor",
        severity_conf=0.85,
        location_conf=0.90,
        cost_max=18000.0,
    )
    assert route == "FAST_TRACK_ELIGIBLE"
    assert "acceptable_evidence_quality" in reasons
    assert "low_fraud_risk" in reasons
    assert "damage_within_limits" in reasons
    assert "cost_within_limits" in reasons
