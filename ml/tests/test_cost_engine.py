"""Unit tests for repair and replacement cost engine."""

from __future__ import annotations

import pytest

from claimvision_ml.costing import (
    CostBreakdownItem,
    CostEstimate,
    estimate_cost,
    normalize_part_name,
)


def test_normalize_part_name():
    """Verify part name normalization handles various input styles."""
    assert normalize_part_name("Front Bumper") == "front_bumper"
    assert normalize_part_name("rear-bumper") == "rear_bumper"
    assert normalize_part_name("HOOD") == "hood"
    assert normalize_part_name("door") == "door"
    assert normalize_part_name("headlamp") == "headlamp"
    assert normalize_part_name("unknown_scratch") == "damage"


def test_estimate_cost_single_part_default():
    """Verify single part cost estimation with default compact segment."""
    est = estimate_cost("front_bumper", severity="minor", segment="compact")

    assert isinstance(est, CostEstimate)
    assert est.currency == "INR"
    assert est.min_cost > 0
    assert est.max_cost >= est.min_cost
    assert len(est.breakdown) == 1
    assert est.breakdown[0].part == "front_bumper"
    assert est.breakdown[0].action == "repair"


def test_estimate_cost_deduplication():
    """Verify repeated parts in the list are deduplicated to prevent double-charging."""
    parts = ["front_bumper", "front bumper", "front-bumper"]
    est = estimate_cost(parts, severity="moderate")

    assert len(est.breakdown) == 1
    assert est.breakdown[0].part == "front_bumper"


def test_estimate_cost_multi_part():
    """Verify multi-part cost estimation aggregates total costs."""
    parts = ["front_bumper", "headlamp"]
    est = estimate_cost(parts, severity="moderate")

    assert len(est.breakdown) == 2
    sum_min = sum(item.min_cost for item in est.breakdown)
    sum_max = sum(item.max_cost for item in est.breakdown)
    assert abs(est.min_cost - sum_min) < 1e-2
    assert abs(est.max_cost - sum_max) < 1e-2


def test_estimate_cost_segment_multipliers():
    """Verify luxury segment multiplier increases cost relative to compact."""
    est_compact = estimate_cost("hood", severity="severe", segment="compact")
    est_luxury = estimate_cost("hood", severity="severe", segment="luxury")

    assert est_luxury.multiplier > est_compact.multiplier
    assert est_luxury.min_cost > est_compact.min_cost
    assert est_luxury.max_cost > est_compact.max_cost


def test_estimate_cost_fallback_empty_parts():
    """Verify empty parts list falls back to generic damage."""
    est = estimate_cost([], severity="minor")
    assert len(est.breakdown) == 1
    assert est.breakdown[0].part == "damage"


def test_cost_estimate_to_dict():
    """Verify serialization to dictionary produces all required keys."""
    est = estimate_cost(["door"], severity="moderate")
    d = est.to_dict()

    assert "currency" in d
    assert "min_cost" in d
    assert "max_cost" in d
    assert "breakdown" in d
    assert "vehicle_segment" in d
    assert isinstance(d["breakdown"], list)
    assert len(d["breakdown"]) == 1
    assert d["breakdown"][0]["part"] == "door"
