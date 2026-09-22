"""costing — rule-based repair-cost estimation engine.

Task ID : INF-001
Phase   : 12 (Unified Inference)

Public API:
    estimate_cost       — compute itemized repair cost range
    CostEstimate        — aggregated estimate dataclass
    CostBreakdownItem   — itemized per-part cost dataclass
"""

from claimvision_ml.costing.cost_engine import (
    CostBreakdownItem,
    CostEstimate,
    estimate_cost,
    normalize_part_name,
)

__all__ = [
    "estimate_cost",
    "CostEstimate",
    "CostBreakdownItem",
    "normalize_part_name",
]
