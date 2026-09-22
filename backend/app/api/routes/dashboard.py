"""dashboard.py — Aggregated triage KPIs and chart analytics endpoint.

Task ID : BE-001
Phase   : 13 (Backend Foundation & Assessment APIs)
"""

from fastapi import APIRouter
from backend.app.schemas.dashboard import DashboardSummary
from backend.app.store import store

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=DashboardSummary)
def get_dashboard_summary() -> DashboardSummary:
    """Retrieve aggregated claim distributions, cost averages, and triage KPIs."""
    summary_data = store.get_dashboard_summary()
    return DashboardSummary(**summary_data)
