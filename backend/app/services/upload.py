"""upload.py — Safe image upload handling and security validation.

Task ID : BE-001
Phase   : 13 (Backend Foundation & Assessment APIs)
Owner   : Member 5 (Backend)

Enforces strict validation rules before persisting files:
1. Whitelisted MIME types (image/jpeg, image/png, image/webp).
2. Maximum payload size (10 MB).
3. Image decode verification via PIL.
4. Minimum resolution check (>= 224x224).
5. SHA-256 fingerprinting for duplicate detection.
"""

from __future__ import annotations

import hashlib
import io
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MIN_WIDTH = 224
MIN_HEIGHT = 224

# Base uploads directory
UPLOADS_DIR = Path(__file__).resolve().parents[3] / "uploads"


def ensure_uploads_dir() -> Path:
    """Ensure the uploads directory exists."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOADS_DIR


async def validate_and_save_upload(
    upload_file: UploadFile,
    claim_id: str,
) -> tuple[str, int, str, str, int, int]:
    """Validate uploaded image file and save to local disk.

    Parameters
    ----------
    upload_file:
        FastAPI UploadFile object.
    claim_id:
        Associated claim identifier.

    Returns
    -------
    tuple:
        (local_path_str, file_size_bytes, mime_type, sha256_hex, width, height)

    Raises
    ------
    HTTPException (422)
        If file fails MIME, extension, size, decode, or resolution check.
    """
    filename = upload_file.filename or "evidence.jpg"
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file extension: '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    content_type = upload_file.content_type or "application/octet-stream"
    if content_type.lower() not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported MIME type: '{content_type}'. Allowed: {sorted(ALLOWED_MIME_TYPES)}",
        )

    # Read content
    content = await upload_file.read()
    file_size = len(content)

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty (0 bytes).",
        )

    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"File exceeds maximum allowed size of 10MB ({file_size} bytes).",
        )

    # Compute SHA-256
    sha256_hash = hashlib.sha256(content).hexdigest()

    # PIL decode verification
    try:
        with Image.open(io.BytesIO(content)) as pil_img:
            width, height = pil_img.size
            # Force verification of image data
            pil_img.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image data is corrupt or cannot be decoded: {exc}",
        ) from exc

    if width < MIN_WIDTH or height < MIN_HEIGHT:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image resolution {width}x{height} is below minimum requirement {MIN_WIDTH}x{MIN_HEIGHT}.",
        )

    # Save to disk
    out_dir = ensure_uploads_dir() / claim_id
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_filename = f"{uuid.uuid4()}{ext}"
    target_path = out_dir / saved_filename

    with open(target_path, "wb") as f:
        f.write(content)

    return (
        str(target_path),
        file_size,
        content_type,
        sha256_hash,
        width,
        height,
    )
