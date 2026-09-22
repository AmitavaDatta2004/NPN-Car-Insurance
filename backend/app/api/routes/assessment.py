"""assessment.py — Assessment orchestration and status polling endpoints.

Task ID : BE-001
Phase   : 13 (Backend Foundation & Assessment APIs)
"""

from fastapi import APIRouter, HTTPException, status
from claimvision_ml.pipeline import assess_claim
from backend.app.store import ClaimStatus, store

router = APIRouter(tags=["assessment"])


@router.post("/claims/{claim_id}/assess", status_code=status.HTTP_200_OK)
def trigger_claim_assessment(claim_id: str) -> dict:
    """Trigger the unified ML assessment pipeline for a claim's evidence."""
    claim = store.get_claim(claim_id)
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim '{claim_id}' not found.",
        )
    if len(claim.images) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Cannot assess claim without any uploaded evidence images.",
        )

    # Transition to assessing stage
    try:
        store.transition_status(
            claim_id,
            to_status=ClaimStatus.ASSESSING_FRAUD,
            reason="Triggered automated assessment pipeline",
            actor="orchestrator",
        )
    except Exception:
        # Proceed even if already in evaluating state
        pass

    # Select primary image for evaluation
    primary_image_path = claim.images[0].local_path

    # Run the unified ML inference orchestrator
    result = assess_claim(
        image_path=primary_image_path,
        claim_id=claim_id,
        vehicle_segment=claim.vehicle_segment,
    )

    # Save assessment and update claim status based on routing decision
    updated_claim = store.save_assessment(claim_id, result.to_dict())

    return updated_claim.assessment or result.to_dict()


@router.get("/assessments/{claim_id}/status", status_code=status.HTTP_200_OK)
def get_assessment_status(claim_id: str) -> dict:
    """Poll processing stage and status for live progress indicator."""
    claim = store.get_claim(claim_id)
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim '{claim_id}' not found.",
        )
    return {
        "claim_id": claim.id,
        "status": claim.status.value,
        "is_complete": claim.status in (
            ClaimStatus.FAST_TRACK_ELIGIBLE,
            ClaimStatus.FRAUD_REVIEW,
            ClaimStatus.MANUAL_DAMAGE_REVIEW,
            ClaimStatus.MORE_EVIDENCE_REQUIRED,
            ClaimStatus.TECHNICAL_REVIEW,
            ClaimStatus.COMPLETED,
        ),
        "has_assessment": claim.assessment is not None,
        "updated_at": claim.updated_at,
    }


@router.get("/claims/{claim_id}/assessment", status_code=status.HTTP_200_OK)
def get_claim_assessment(claim_id: str) -> dict:
    """Retrieve full structured assessment report for a claim."""
    claim = store.get_claim(claim_id)
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim '{claim_id}' not found.",
        )
    if not claim.assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim '{claim_id}' has not been assessed yet.",
        )
    return claim.assessment
