"""Deterministic OpenCV image quality and evidence checks.

Provides safe image loading, decodability verification, dimension extraction,
Laplacian blur scoring, brightness calculation, and contrast calculation.
Preserves original image data without silent modification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_image_safely(
    image_path: str | Path,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Attempt to decode an image with OpenCV safely.

    Args:
        image_path: Path to the image file.

    Returns:
        A tuple of (bgr_image, metadata_dict). If unreadable or corrupt,
        bgr_image is None and metadata_dict contains error information.
    """
    path_str = str(image_path)
    meta: dict[str, Any] = {
        "path": path_str,
        "is_valid": False,
        "width": 0,
        "height": 0,
        "channels": 0,
        "aspect_ratio": 0.0,
        "error": None,
    }

    p = Path(image_path)
    if not p.exists() or not p.is_file():
        meta["error"] = "file_not_found"
        return None, meta

    if p.stat().st_size == 0:
        meta["error"] = "empty_file"
        return None, meta

    try:
        # cv2.imread returns None on corrupt or unsupported images
        image = cv2.imread(path_str, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            meta["error"] = "corrupt_or_undecodable"
            return None, meta

        h, w = image.shape[:2]
        c = image.shape[2] if image.ndim == 3 else 1

        meta["is_valid"] = True
        meta["width"] = int(w)
        meta["height"] = int(h)
        meta["channels"] = int(c)
        meta["aspect_ratio"] = round(float(w) / float(h), 4) if h > 0 else 0.0
        return image, meta

    except Exception as exc:
        meta["error"] = f"decode_exception: {exc}"
        return None, meta


def calculate_blur_score(image: np.ndarray) -> float:
    """Calculate the blur score using variance of the Laplacian on grayscale.

    Higher variance corresponds to sharper edges; lower variance indicates blur.
    """
    if image is None or image.size == 0:
        return 0.0

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())
    return round(variance, 4)


def calculate_brightness(image: np.ndarray) -> float:
    """Calculate mean brightness of the image (0.0 to 255.0)."""
    if image is None or image.size == 0:
        return 0.0

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    return round(float(np.mean(gray)), 4)


def calculate_contrast(image: np.ndarray) -> float:
    """Calculate contrast as the standard deviation of grayscale intensity."""
    if image is None or image.size == 0:
        return 0.0

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    return round(float(np.std(gray)), 4)


def compute_image_metrics(image_path: str | Path) -> dict[str, Any]:
    """Inspect an image and return complete quality metrics dictionary.

    Returns:
        Dictionary with status, dimensions, blur_score, brightness, and contrast.
    """
    image, meta = read_image_safely(image_path)
    if not meta["is_valid"] or image is None:
        meta.update(
            {
                "blur_score": 0.0,
                "brightness": 0.0,
                "contrast": 0.0,
            }
        )
        return meta

    meta["blur_score"] = calculate_blur_score(image)
    meta["brightness"] = calculate_brightness(image)
    meta["contrast"] = calculate_contrast(image)
    return meta


# ---------------------------------------------------------------------------
# Runtime checks added in Phase 3
# ---------------------------------------------------------------------------


def check_minimum_resolution(
    image: np.ndarray,
    min_width: int = 224,
    min_height: int = 224,
) -> tuple[bool, str]:
    """Check whether an image meets the minimum resolution requirement.

    Args:
        image: BGR image array already loaded by read_image_safely.
        min_width: Minimum required width in pixels.
        min_height: Minimum required height in pixels.

    Returns:
        Tuple of (passed, reason_code).
        passed=True means the image meets the minimum resolution.
        reason_code is empty string on pass, or "resolution_too_low" on fail.
    """
    if image is None or image.size == 0:
        return False, "resolution_too_low"

    h, w = image.shape[:2]
    if w < min_width or h < min_height:
        return False, "resolution_too_low"
    return True, ""


def correct_orientation(
    image_path: str | Path,
) -> tuple[np.ndarray | None, bool, list[str]]:
    """Correct image orientation using EXIF tag via Pillow.

    Reads the EXIF orientation tag if present and rotates the image to its
    upright position. If EXIF data is absent or the orientation tag is missing,
    the original image is returned unchanged and a warning is appended.

    IMPORTANT: EXIF absence alone is not a fraud or quality failure.
    It adds a non-fatal warning only. See README §12.

    Args:
        image_path: Path to the image file.

    Returns:
        Tuple of (bgr_image, orientation_corrected, warnings).
        bgr_image: Orientation-corrected BGR numpy array, or None if unreadable.
        orientation_corrected: True if a rotation was actually applied.
        warnings: List of warning strings (may contain "exif_absent").
    """
    from PIL import Image as PilImage
    from PIL.ExifTags import TAGS

    warnings: list[str] = []
    path = Path(image_path)

    try:
        pil_img = PilImage.open(path)
    except Exception:
        return None, False, ["image_open_failed"]

    exif_data = None
    try:
        exif_data = pil_img._getexif()  # type: ignore[attr-defined]
    except Exception:
        pass

    if exif_data is None:
        warnings.append("exif_absent")
        # Return the image as-is; absence of EXIF is not a quality failure
        rgb = np.array(pil_img.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr, False, warnings

    # Find the orientation tag value
    orientation_key = next(
        (k for k, v in TAGS.items() if v == "Orientation"), None
    )
    orientation_value = exif_data.get(orientation_key) if orientation_key else None

    if orientation_value is None:
        warnings.append("exif_absent")
        rgb = np.array(pil_img.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr, False, warnings

    # Apply rotation based on EXIF orientation value
    _orientation_to_rotation = {
        3: PilImage.ROTATE_180,
        6: PilImage.ROTATE_270,
        8: PilImage.ROTATE_90,
    }
    rotation = _orientation_to_rotation.get(orientation_value)
    if rotation is not None:
        pil_img = pil_img.transpose(rotation)
        orientation_corrected = True
    else:
        orientation_corrected = False

    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, orientation_corrected, warnings


def extract_exif_summary(image_path: str | Path) -> dict[str, Any]:
    """Extract a safe, non-private subset of EXIF metadata from an image.

    Returns a dictionary of selected EXIF fields. Does NOT expose GPS
    coordinates or personal identifying data.

    IMPORTANT: exif_available=False is a non-fatal informational flag only.
    It must never be used as a fraud or quality rejection signal. See README §12.

    Args:
        image_path: Path to the image file.

    Returns:
        Dictionary with:
            exif_available (bool): True if any EXIF data was found.
            make (str | None): Camera manufacturer.
            model (str | None): Camera model.
            datetime (str | None): Original capture datetime string.
            has_gps (bool): True if GPS data is present (coordinates not returned).
            orientation (int | None): Raw EXIF orientation value.
    """
    from PIL import Image as PilImage
    from PIL.ExifTags import TAGS

    summary: dict[str, Any] = {
        "exif_available": False,
        "make": None,
        "model": None,
        "datetime": None,
        "has_gps": False,
        "orientation": None,
    }

    try:
        pil_img = PilImage.open(image_path)
        exif_data = pil_img._getexif()  # type: ignore[attr-defined]
    except Exception:
        return summary

    if exif_data is None:
        return summary

    summary["exif_available"] = True
    tag_lookup = {v: k for k, v in TAGS.items()}

    for tag_name, result_key in [
        ("Make", "make"),
        ("Model", "model"),
        ("DateTimeOriginal", "datetime"),
    ]:
        tag_id = tag_lookup.get(tag_name)
        if tag_id and tag_id in exif_data:
            summary[result_key] = str(exif_data[tag_id])

    gps_tag_id = tag_lookup.get("GPSInfo")
    if gps_tag_id and gps_tag_id in exif_data:
        summary["has_gps"] = True

    orientation_tag_id = tag_lookup.get("Orientation")
    if orientation_tag_id and orientation_tag_id in exif_data:
        summary["orientation"] = exif_data[orientation_tag_id]

    return summary


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    labels: list[str] | None = None,
    scores: list[float] | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    """Draw bounding boxes and optional labels/scores on a copy of the image.

    Args:
        image: BGR numpy array (original is not modified).
        boxes: List of (x1, y1, x2, y2) bounding boxes in pixel coordinates.
        labels: Optional list of label strings, one per box.
        scores: Optional list of confidence floats, one per box.
        color: BGR colour tuple for the box outline.
        thickness: Line thickness in pixels.
        font_scale: Font size for the label overlay.

    Returns:
        New BGR numpy array with boxes and labels drawn.
    """
    annotated = image.copy()
    if labels is None:
        labels = [""] * len(boxes)
    if scores is None:
        scores = [None] * len(boxes)  # type: ignore[list-item]

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        text_parts = []
        if label:
            text_parts.append(label)
        if score is not None:
            text_parts.append(f"{score:.2f}")
        text = " ".join(text_parts)

        if text:
            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            bg_y1 = max(y1 - text_h - baseline - 4, 0)
            cv2.rectangle(
                annotated,
                (x1, bg_y1),
                (x1 + text_w + 4, y1),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                annotated,
                text,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    return annotated


def save_annotated_image(
    image: np.ndarray,
    output_path: str | Path,
) -> str:
    """Save an annotated BGR image to the specified output path.

    Creates parent directories if they do not exist.

    Args:
        image: BGR numpy array to save.
        output_path: Destination file path (should be inside `annotated/`).

    Returns:
        Absolute path string of the saved file.

    Raises:
        RuntimeError: If cv2.imwrite fails.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(out), image)
    if not success:
        raise RuntimeError(f"cv2.imwrite failed for path: {out}")
    return str(out.resolve())
