"""claims.py — Claim draft lifecycle, submission, and timeline endpoints.

Task ID : BE-001
Phase   : 13 (Backend Foundation & Assessment APIs)
"""

from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.claim import (
    ClaimCreate,
    ClaimRead,
    ClaimUpdate,
    TimelineEventRead,
)
from backend.app.store import ClaimStatus, store

router = APIRouter(prefix="/claims", tags=["claims"])


@router.post("", response_model=ClaimRead, status_code=status.HTTP_201_CREATED)
def create_claim(payload: ClaimCreate) -> ClaimRead:
    """Create a new draft claim."""
    claim = store.create_claim(
        policy_number=payload.policy_number,
        vehicle_make=payload.vehicle_make,
        vehicle_model=payload.vehicle_model,
        vehicle_year=payload.vehicle_year,
        vehicle_segment=payload.vehicle_segment,
        incident_description=payload.incident_description,
        incident_date=payload.incident_date,
    )
    return ClaimRead(**claim.to_dict())


@router.get("", response_model=list[ClaimRead])
def list_claims() -> list[ClaimRead]:
    """List all claims ordered by creation timestamp."""
    return [ClaimRead(**c.to_dict()) for c in store.list_claims()]


@router.get("/{claim_id}", response_model=ClaimRead)
def get_claim(claim_id: str) -> ClaimRead:
    """Fetch claim details by ID."""
    claim = store.get_claim(claim_id)
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim '{claim_id}' not found.",
        )
    return ClaimRead(**claim.to_dict())


@router.patch("/{claim_id}", response_model=ClaimRead)
def update_claim(claim_id: str, payload: ClaimUpdate) -> ClaimRead:
    """Update metadata for an existing draft claim."""
    claim = store.get_claim(claim_id)
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim '{claim_id}' not found.",
        )
    if claim.status != ClaimStatus.DRAFT:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Only DRAFT claims can be modified; current status is '{claim.status.value}'.",
        )

    updated = store.update_claim(claim_id, **payload.model_dump(exclude_unset=True))
    return ClaimRead(**updated.to_dict())


@router.post("/{claim_id}/submit", response_model=ClaimRead)
def submit_claim(claim_id: str) -> ClaimRead:
    """Lock draft claim and transition state to SUBMITTED."""
    claim = store.get_claim(claim_id)
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim '{claim_id}' not found.",
        )
    if len(claim.images) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Cannot submit claim with zero evidence images. Please upload at least one image.",
        )

    try:
        updated = store.transition_status(
            claim_id,
            to_status=ClaimStatus.SUBMITTED,
            reason="Policyholder submitted claim evidence",
            actor="policyholder",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return ClaimRead(**updated.to_dict())


@router.get("/{claim_id}/timeline", response_model=list[TimelineEventRead])
def get_claim_timeline(claim_id: str) -> list[TimelineEventRead]:
    """Retrieve full audit history of state transitions for a claim."""
    claim = store.get_claim(claim_id)
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim '{claim_id}' not found.",
        )
    return [TimelineEventRead(**ev.to_dict()) for ev in claim.timeline]
