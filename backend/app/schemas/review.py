"""review.py — Pydantic schemas for adjuster reviews and decisions.

Task ID : BE-001
Phase   : 13 (Backend Foundation & Assessment APIs)
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class ReviewCorrection(BaseModel):
    """Payload to save intermediate adjuster findings or notes."""

    model_config = ConfigDict(extra="ignore")

    reviewer_id: str = Field("adjuster_1", examples=["adjuster_1"])
    notes: str = Field("", max_length=5000)
    overrides: dict[str, Any] = Field(
        default_factory=dict,
        examples=[{"severity": "moderate", "parts": ["front_bumper", "headlamp"]}],
    )


class ReviewDecision(BaseModel):
    """Payload to finalize an adjuster review decision."""

    model_config = ConfigDict(extra="ignore")

    reviewer_id: str = Field("adjuster_1", examples=["adjuster_1"])
    decision: str = Field(..., examples=["APPROVED", "REJECTED", "REQUEST_INFO"])
    notes: str = Field("", max_length=5000)
    overrides: dict[str, Any] | None = None


class ReviewRead(BaseModel):
    """Response shape for an adjuster review record."""

    reviewer_id: str
    decision: str
    notes: str
    overrides: dict[str, Any]
    timestamp: str
