"""dashboard.py — Pydantic schemas for dashboard KPIs and chart data.

Task ID : BE-001
Phase   : 13 (Backend Foundation & Assessment APIs)
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class CostRange(BaseModel):
    min: float
    max: float


class DashboardSummary(BaseModel):
    """Aggregate triage KPIs and distributions for the reviewer dashboard."""

    total_claims: int
    status_counts: dict[str, int]
    fast_track_count: int
    fraud_review_count: int
    damage_review_count: int
    completed_count: int
    average_cost: CostRange
    severity_distribution: dict[str, int]
    location_distribution: dict[str, int]
    fraud_score_distribution: list[float]
    override_rate: float
