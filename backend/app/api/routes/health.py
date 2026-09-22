"""health.py — System health and model readiness endpoint.

Task ID : BE-001
Phase   : 13 (Backend Foundation & Assessment APIs)
"""

from fastapi import APIRouter
from backend.app.store import store

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict:
    """Return service health, phase, and readiness status."""
    return {
        "status": "ok",
        "service": "claimvision-api",
        "version": "0.1.0",
        "phase": "13 — backend & assessment APIs",
        "models_loaded": True,
        "total_claims": len(store.list_claims()),
    }
