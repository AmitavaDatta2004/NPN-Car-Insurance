"""images.py — Evidence image upload and deletion endpoints.

Task ID : BE-001
Phase   : 13 (Backend Foundation & Assessment APIs)
"""

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from backend.app.schemas.claim import ImageRead
from backend.app.services.upload import validate_and_save_upload
from backend.app.store import ClaimStatus, store

router = APIRouter(tags=["images"])


@router.post(
    "/claims/{claim_id}/images",
    response_model=ImageRead,
    status_code=status.HTTP_201_CREATED,
)
async def upload_claim_image(
    claim_id: str,
    file: UploadFile = File(...),
) -> ImageRead:
    """Upload and attach a validated evidence photograph to a claim."""
    claim = store.get_claim(claim_id)
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim '{claim_id}' not found.",
        )
    if claim.status not in (ClaimStatus.DRAFT, ClaimStatus.MORE_EVIDENCE_REQUIRED):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Images can only be uploaded when status is DRAFT or MORE_EVIDENCE_REQUIRED (current: '{claim.status.value}').",
        )

    # Validate and persist to uploads/
    (
        local_path,
        file_size,
        mime_type,
        sha256_hash,
        width,
        height,
    ) = await validate_and_save_upload(file, claim_id)

    record = store.add_image(
        claim_id=claim_id,
        filename=file.filename or "evidence.jpg",
        local_path=local_path,
        file_size_bytes=file_size,
        mime_type=mime_type,
        sha256=sha256_hash,
        width=width,
        height=height,
    )

    return ImageRead(**record.to_dict())


@router.delete("/images/{image_id}", status_code=status.HTTP_200_OK)
def delete_image(image_id: str) -> dict:
    """Remove an evidence image from a draft claim."""
    img = store.get_image(image_id)
    if not img:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image '{image_id}' not found.",
        )
    claim = store.get_claim(img.claim_id)
    if claim and claim.status != ClaimStatus.DRAFT:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Images can only be removed from DRAFT claims.",
        )

    deleted = store.delete_image(image_id)
    return {"deleted": deleted, "image_id": image_id}
