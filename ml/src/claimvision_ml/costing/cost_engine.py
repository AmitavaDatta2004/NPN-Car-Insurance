"""cost_engine.py — Rule-based repair and replacement cost estimation.

Task ID : INF-001
Phase   : 12 (Unified Inference)
Owner   : Member 5 (Backend/ML)

Features:
- Lookup based on part, damage severity, and vehicle segment.
- Versioned JSON rates table in config/cost_table.json.
- Deduplication of detected parts to avoid double-charging.
- Returns comprehensive CostEstimate with itemized breakdown.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Default fallback rates if config/cost_table.json cannot be read
DEFAULT_RATES: dict[str, dict[str, dict[str, Any]]] = {
    "front_bumper": {
        "minor": {"action": "repair", "part_min": 2000, "part_max": 4000, "labor_min": 2500, "labor_max": 4500, "paint_min": 3000, "paint_max": 5000},
        "moderate": {"action": "repair", "part_min": 5000, "part_max": 9000, "labor_min": 4000, "labor_max": 7000, "paint_min": 4000, "paint_max": 7000},
        "severe": {"action": "replace", "part_min": 14000, "part_max": 24000, "labor_min": 6000, "labor_max": 10000, "paint_min": 6000, "paint_max": 10000},
    },
    "rear_bumper": {
        "minor": {"action": "repair", "part_min": 2000, "part_max": 4000, "labor_min": 2500, "labor_max": 4500, "paint_min": 3000, "paint_max": 5000},
        "moderate": {"action": "repair", "part_min": 5000, "part_max": 9000, "labor_min": 4000, "labor_max": 7000, "paint_min": 4000, "paint_max": 7000},
        "severe": {"action": "replace", "part_min": 15000, "part_max": 26000, "labor_min": 6000, "labor_max": 10000, "paint_min": 6000, "paint_max": 10000},
    },
    "hood": {
        "minor": {"action": "repair", "part_min": 3000, "part_max": 6000, "labor_min": 3500, "labor_max": 6000, "paint_min": 4500, "paint_max": 7500},
        "moderate": {"action": "repair", "part_min": 7000, "part_max": 12000, "labor_min": 5500, "labor_max": 9500, "paint_min": 6000, "paint_max": 10000},
        "severe": {"action": "replace", "part_min": 20000, "part_max": 38000, "labor_min": 8000, "labor_max": 14000, "paint_min": 8000, "paint_max": 14000},
    },
    "door": {
        "minor": {"action": "repair", "part_min": 2500, "part_max": 5000, "labor_min": 3000, "labor_max": 5000, "paint_min": 4000, "paint_max": 6500},
        "moderate": {"action": "repair", "part_min": 6000, "part_max": 11000, "labor_min": 5000, "labor_max": 8500, "paint_min": 5500, "paint_max": 9000},
        "severe": {"action": "replace", "part_min": 18000, "part_max": 32000, "labor_min": 7000, "labor_max": 12000, "paint_min": 7000, "paint_max": 12000},
    },
    "headlamp": {
        "minor": {"action": "repair", "part_min": 1000, "part_max": 2500, "labor_min": 1500, "labor_max": 2500, "paint_min": 0, "paint_max": 0},
        "moderate": {"action": "replace", "part_min": 6000, "part_max": 12000, "labor_min": 2000, "labor_max": 3500, "paint_min": 0, "paint_max": 0},
        "severe": {"action": "replace", "part_min": 12000, "part_max": 25000, "labor_min": 2500, "labor_max": 4500, "paint_min": 0, "paint_max": 0},
    },
    "damage": {
        "minor": {"action": "repair", "part_min": 2000, "part_max": 4000, "labor_min": 2500, "labor_max": 4500, "paint_min": 3000, "paint_max": 5000},
        "moderate": {"action": "repair", "part_min": 5000, "part_max": 10000, "labor_min": 4500, "labor_max": 8000, "paint_min": 4500, "paint_max": 8000},
        "severe": {"action": "replace", "part_min": 16000, "part_max": 30000, "labor_min": 7000, "labor_max": 12000, "paint_min": 7000, "paint_max": 12000},
    },
}

DEFAULT_SEGMENT_MULTIPLIERS: dict[str, float] = {
    "hatchback": 0.85,
    "compact": 1.0,
    "sedan": 1.15,
    "suv": 1.35,
    "luxury": 2.2,
}


@dataclass
class CostBreakdownItem:
    """Itemized repair/replace cost for a single damaged part."""

    part: str
    severity: str
    action: str
    min_cost: float
    max_cost: float
    part_min: float = 0.0
    part_max: float = 0.0
    labor_min: float = 0.0
    labor_max: float = 0.0
    paint_min: float = 0.0
    paint_max: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize item to dictionary."""
        return asdict(self)


@dataclass
class CostEstimate:
    """Aggregated repair and replacement cost estimate."""

    currency: str
    min_cost: float
    max_cost: float
    breakdown: list[CostBreakdownItem] = field(default_factory=list)
    vehicle_segment: str = "compact"
    multiplier: float = 1.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize estimate to dictionary."""
        return {
            "currency": self.currency,
            "min_cost": round(self.min_cost, 2),
            "max_cost": round(self.max_cost, 2),
            "vehicle_segment": self.vehicle_segment,
            "multiplier": self.multiplier,
            "breakdown": [item.to_dict() for item in self.breakdown],
            "notes": self.notes,
        }


def _load_cost_table(config_path: str | Path | None = None) -> tuple[dict[str, Any], dict[str, float], str]:
    """Load cost table JSON or return default rates."""
    if config_path is not None:
        p = Path(config_path)
    else:
        p = Path(__file__).resolve().parents[4] / "config" / "cost_table.json"

    if p.is_file():
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            rates = data.get("rates", DEFAULT_RATES)
            multipliers = data.get("segment_multipliers", DEFAULT_SEGMENT_MULTIPLIERS)
            currency = data.get("currency", "INR")
            return rates, multipliers, currency
        except Exception:
            pass

    return DEFAULT_RATES, DEFAULT_SEGMENT_MULTIPLIERS, "INR"


def normalize_part_name(part_name: str) -> str:
    """Normalize user or model part names to standard keys."""
    cleaned = part_name.strip().lower().replace(" ", "_").replace("-", "_")
    valid_parts = {"front_bumper", "rear_bumper", "hood", "door", "headlamp", "damage"}
    if cleaned in valid_parts:
        return cleaned
    # Substring matching for convenience
    for vp in ["front_bumper", "rear_bumper", "headlamp", "hood", "door"]:
        if vp in cleaned:
            return vp
    return "damage"


def estimate_cost(
    parts: list[str] | str,
    severity: str = "moderate",
    segment: str = "compact",
    config_path: str | Path | None = None,
) -> CostEstimate:
    """Estimate vehicle repair costs using rule-based lookup table.

    Parameters
    ----------
    parts:
        List of damaged parts (e.g. ['front_bumper', 'headlamp']) or single string.
    severity:
        Damage severity tier: 'minor', 'moderate', or 'severe'.
    segment:
        Vehicle segment: 'hatchback', 'compact', 'sedan', 'suv', 'luxury'.
    config_path:
        Optional path to custom cost_table.json.

    Returns
    -------
    CostEstimate
        Structured estimate with min/max cost and itemized breakdown.
    """
    rates, multipliers, currency = _load_cost_table(config_path)

    # Normalize severity
    norm_sev = severity.strip().lower()
    if norm_sev not in ("minor", "moderate", "severe"):
        norm_sev = "moderate"

    # Normalize segment & multiplier
    norm_seg = segment.strip().lower()
    mult = multipliers.get(norm_seg, 1.0)

    # Normalize parts list and deduplicate
    if isinstance(parts, str):
        parts_list = [parts] if parts.strip() else []
    else:
        parts_list = list(parts)

    norm_parts: list[str] = []
    seen: set[str] = set()
    for p in parts_list:
        np_name = normalize_part_name(p)
        if np_name not in seen:
            seen.add(np_name)
            norm_parts.append(np_name)

    if not norm_parts:
        norm_parts = ["damage"]

    breakdown: list[CostBreakdownItem] = []
    total_min = 0.0
    total_max = 0.0

    for part_name in norm_parts:
        part_rates = rates.get(part_name, rates.get("damage", DEFAULT_RATES["damage"]))
        tier_rates = part_rates.get(norm_sev, part_rates.get("moderate", DEFAULT_RATES["damage"]["moderate"]))

        action = tier_rates.get("action", "repair")
        p_min = tier_rates.get("part_min", 0.0) * mult
        p_max = tier_rates.get("part_max", 0.0) * mult
        l_min = tier_rates.get("labor_min", 0.0) * mult
        l_max = tier_rates.get("labor_max", 0.0) * mult
        pt_min = tier_rates.get("paint_min", 0.0) * mult
        pt_max = tier_rates.get("paint_max", 0.0) * mult

        item_min = p_min + l_min + pt_min
        item_max = p_max + l_max + pt_max

        total_min += item_min
        total_max += item_max

        breakdown.append(
            CostBreakdownItem(
                part=part_name,
                severity=norm_sev,
                action=action,
                min_cost=round(item_min, 2),
                max_cost=round(item_max, 2),
                part_min=round(p_min, 2),
                part_max=round(p_max, 2),
                labor_min=round(l_min, 2),
                labor_max=round(l_max, 2),
                paint_min=round(pt_min, 2),
                paint_max=round(pt_max, 2),
            )
        )

    notes = [
        f"Segment multiplier applied: {mult:.2f} ({norm_seg})",
        "Illustrative prototype estimation; subject to physical adjuster appraisal.",
    ]

    return CostEstimate(
        currency=currency,
        min_cost=round(total_min, 2),
        max_cost=round(total_max, 2),
        breakdown=breakdown,
        vehicle_segment=norm_seg,
        multiplier=mult,
        notes=notes,
    )
