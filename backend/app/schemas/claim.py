"""claim.py — Pydantic schemas for claims and images.

Task ID : BE-001
Phase   : 13 (Backend Foundation & Assessment APIs)
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class ClaimCreate(BaseModel):
    """Payload to initiate a new claim draft."""

    model_config = ConfigDict(extra="ignore")

    policy_number: str = Field(..., min_length=3, max_length=50, examples=["POL-987654"])
    vehicle_make: str = Field("Toyota", examples=["Toyota"])
    vehicle_model: str = Field("Corolla", examples=["Corolla"])
    vehicle_year: int = Field(2022, ge=1990, le=2030, examples=[2022])
    vehicle_segment: str = Field("compact", examples=["compact", "sedan", "suv"])
    incident_description: str = Field("", max_length=2000)
    incident_date: str = Field("", examples=["2026-09-20"])


class ClaimUpdate(BaseModel):
    """Payload to update an existing draft claim."""

    model_config = ConfigDict(extra="ignore")

    policy_number: str | None = None
    vehicle_make: str | None = None
    vehicle_model: str | None = None
    vehicle_year: int | None = None
    vehicle_segment: str | None = None
    incident_description: str | None = None
    incident_date: str | None = None


class ImageRead(BaseModel):
    """Response shape for an uploaded evidence image."""

    id: str
    claim_id: str
    filename: str
    local_path: str
    file_size_bytes: int
    mime_type: str
    sha256: str
    width: int
    height: int
    created_at: str


class TimelineEventRead(BaseModel):
    """Response shape for a claim status transition event."""

    id: str
    claim_id: str
    from_status: str
    to_status: str
    reason: str
    actor: str
    timestamp: str


class ClaimRead(BaseModel):
    """Full claim response including evidence images, assessment, and timeline."""

    id: str
    policy_number: str
    vehicle_make: str
    vehicle_model: str
    vehicle_year: int
    vehicle_segment: str
    incident_description: str
    incident_date: str
    status: str
    images: list[ImageRead] = Field(default_factory=list)
    assessment: dict[str, Any] | None = None
    review: dict[str, Any] | None = None
    timeline: list[TimelineEventRead] = Field(default_factory=list)
    created_at: str
    updated_at: str
