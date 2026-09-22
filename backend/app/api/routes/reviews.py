"""reviews.py — Adjuster review queue, manual corrections, and final decision endpoints.

Task ID : BE-001
Phase   : 13 (Backend Foundation & Assessment APIs)
"""

from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.claim import ClaimRead
from backend.app.schemas.review import ReviewCorrection, ReviewDecision
from backend.app.store import store

router = APIRouter(prefix="/reviews", tags=["reviews"])


@router.get("/queue", response_model=list[ClaimRead])
def get_review_queue() -> list[ClaimRead]:
    """Retrieve list of claims requiring human adjuster review."""
    queue = store.get_review_queue()
    return [ClaimRead(**c.to_dict()) for c in queue]


@router.patch("/{claim_id}", response_model=ClaimRead)
def save_review_correction(claim_id: str, payload: ReviewCorrection) -> ClaimRead:
    """Save intermediate adjuster observations and manual overrides without overwriting AI output."""
    claim = store.get_claim(claim_id)
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim '{claim_id}' not found.",
        )

    updated = store.update_claim(
        claim_id,
        review={
            "reviewer_id": payload.reviewer_id,
            "decision": "PENDING",
            "notes": payload.notes,
            "overrides": payload.overrides,
        },
    )
    return ClaimRead(**updated.to_dict())


@router.post("/{claim_id}/decision", response_model=ClaimRead)
def submit_review_decision(claim_id: str, payload: ReviewDecision) -> ClaimRead:
    """Submit final review decision (APPROVED/REJECTED) and move claim to COMPLETED."""
    claim = store.get_claim(claim_id)
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim '{claim_id}' not found.",
        )

    updated = store.save_review(
        claim_id=claim_id,
        reviewer_id=payload.reviewer_id,
        decision=payload.decision,
        notes=payload.notes,
        overrides=payload.overrides,
    )
    return ClaimRead(**updated.to_dict())
