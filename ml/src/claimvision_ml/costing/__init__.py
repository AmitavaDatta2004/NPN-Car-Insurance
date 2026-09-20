"""costing — rule-based repair-cost estimation engine.

Implemented in Phase 11–12 (backend integration).
Cost table stored in config/cost_table.json (versioned JSON, not in Git for now).

Planned public API:
    estimate_cost(parts, severity, vehicle_segment) -> CostEstimate
"""
