"""Runtime evidence-integrity orchestrator for Phase 3.

Provides run_quality_checks() — the primary entry point called by the
assessment pipeline before any ML inference.

Every uploaded image must pass through this checker first. The result
determines whether the image is acceptable evidence, needs to be re-submitted,
or matches a known duplicate in the historical store.

Check order (all deterministic, no ML models):
    1. Corrupt / unreadable  → MORE_EVIDENCE_REQUIRED
    2. Minimum resolution    → MORE_EVIDENCE_REQUIRED
    3. Blur score            → rejection_reasons (does not auto-fail alone)
    4. Brightness            → rejection_reasons (too dark or overexposed)
    5. Contrast              → rejection_reasons (too low)
    6. Orientation fix       → silent correction via EXIF (non-fatal if absent)
    7. Exact duplicate       → DUPLICATE_REVIEW
    8. Near-duplicate        → DUPLICATE_REVIEW
    9. EXIF availability     → warning only (NEVER a fraud or quality failure)

IMPORTANT: exif_absent is a non-fatal warning. It must never set passed=False
or route=FRAUD_REVIEW. See README §12.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import imagehash
from PIL import Image as PilImage

from claimvision_ml.quality.image_checks import (
    calculate_blur_score,
    calculate_brightness,
    calculate_contrast,
    check_minimum_resolution,
    correct_orientation,
    extract_exif_summary,
    read_image_safely,
)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class QualityResult:
    """Complete result of run_quality_checks() for one image.

    Attributes:
        image_path: Path to the evaluated image.
        passed: True if the image is acceptable evidence for downstream ML.
        route: Routing decision string.
            "CONTINUE"               → acceptable; proceed to fraud/damage.
            "MORE_EVIDENCE_REQUIRED" → image is corrupt, too small, or fails
                                       enough quality checks to be unusable.
            "DUPLICATE_REVIEW"       → exact or near-duplicate detected;
                                       routes to human review (not FRAUD_REVIEW).
        rejection_reasons: Human-readable reason codes. Multiple reasons may
            appear when several quality checks fail simultaneously.
        warnings: Non-fatal informational flags (e.g. "exif_absent").
            These never change passed or route on their own.
        width: Decoded image width in pixels (0 if unreadable).
        height: Decoded image height in pixels (0 if unreadable).
        blur_score: Laplacian variance (higher = sharper). 0.0 if unreadable.
        brightness: Mean grayscale intensity [0, 255]. 0.0 if unreadable.
        contrast: Standard deviation of grayscale intensity. 0.0 if unreadable.
        sha256: SHA-256 hex digest of the raw file bytes.
        is_exact_duplicate: True if sha256 matches a historical image.
        is_near_duplicate: True if dHash is within threshold of a historical image.
        duplicate_reference: Hash or path of the matched historical image, or None.
        orientation_corrected: True if EXIF rotation was actually applied.
        exif_available: True if EXIF metadata was found (informational only).
        check_ms: Total wall-clock time for all checks in milliseconds.
    """

    image_path: str
    passed: bool
    route: str              # "CONTINUE" | "MORE_EVIDENCE_REQUIRED" | "DUPLICATE_REVIEW"
    rejection_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Dimensions
    width: int = 0
    height: int = 0

    # Quality scores
    blur_score: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0

    # Duplicate signals
    sha256: str = ""
    is_exact_duplicate: bool = False
    is_near_duplicate: bool = False
    duplicate_reference: str | None = None

    # Metadata
    orientation_corrected: bool = False
    exif_available: bool = False

    # Timing
    check_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise result to a plain dictionary (JSON-compatible)."""
        return {
            "image_path": self.image_path,
            "passed": self.passed,
            "route": self.route,
            "rejection_reasons": self.rejection_reasons,
            "warnings": self.warnings,
            "width": self.width,
            "height": self.height,
            "blur_score": self.blur_score,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "sha256": self.sha256,
            "is_exact_duplicate": self.is_exact_duplicate,
            "is_near_duplicate": self.is_near_duplicate,
            "duplicate_reference": self.duplicate_reference,
            "orientation_corrected": self.orientation_corrected,
            "exif_available": self.exif_available,
            "check_ms": self.check_ms,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_sha256(image_path: Path) -> str:
    """Return SHA-256 hex digest of raw file bytes."""
    hasher = hashlib.sha256()
    with open(image_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _compute_dhash(image_path: Path, hash_size: int = 8) -> str | None:
    """Return dHash string for the image, or None on failure."""
    try:
        with PilImage.open(image_path) as img:
            return str(imagehash.dhash(img, hash_size=hash_size))
    except Exception:
        return None


def _hamming_distance(hash_a: str, hash_b: str) -> int:
    """Compute Hamming distance between two hex dHash strings."""
    try:
        h_a = imagehash.hex_to_hash(hash_a)
        h_b = imagehash.hex_to_hash(hash_b)
        return int(h_a - h_b)
    except Exception:
        return 999  # treat as non-duplicate on error


# ---------------------------------------------------------------------------
# Primary public function
# ---------------------------------------------------------------------------


def run_quality_checks(
    image_path: str | Path,
    historical_hashes: dict[str, str] | None = None,
    historical_dhashes: dict[str, str] | None = None,
    blur_threshold: float = 50.0,
    min_brightness: float = 30.0,
    max_brightness: float = 240.0,
    min_contrast: float = 15.0,
    min_width: int = 224,
    min_height: int = 224,
    near_dup_threshold: int = 4,
) -> QualityResult:
    """Run all deterministic evidence-integrity checks on one image.

    This function is the primary runtime entry point called by the assessment
    orchestrator before fraud inference or damage detection.

    Check execution order:
        1. Corrupt / unreadable check
        2. Minimum resolution check
        3. Blur score check
        4. Brightness check (too dark or overexposed)
        5. Contrast check
        6. Orientation correction (via EXIF; silent, non-fatal)
        7. Exact SHA-256 duplicate check
        8. Near-duplicate dHash check
        9. EXIF summary (informational only)

    Args:
        image_path: Path to the image file to evaluate.
        historical_hashes: Optional mapping of sha256_hex → reference_path
            for exact-duplicate detection against historical evidence.
        historical_dhashes: Optional mapping of dhash_str → reference_path
            for near-duplicate detection against historical evidence.
        blur_threshold: Laplacian variance below this → "excessive_blur".
            Default 50.0 (illustrative; not tuned on real validation data).
        min_brightness: Mean grayscale below this → "too_dark".
        max_brightness: Mean grayscale above this → "overexposed".
        min_contrast: Grayscale std dev below this → "low_contrast".
        min_width: Minimum image width in pixels.
        min_height: Minimum image height in pixels.
        near_dup_threshold: Max Hamming distance to consider near-duplicate.

    Returns:
        QualityResult dataclass with all findings, scores, and routing decision.
    """
    t_start = time.perf_counter()
    image_path = Path(image_path)

    result = QualityResult(
        image_path=str(image_path),
        passed=False,
        route="MORE_EVIDENCE_REQUIRED",
    )

    # ------------------------------------------------------------------
    # Check 1: Corrupt / unreadable
    # ------------------------------------------------------------------
    image, meta = read_image_safely(image_path)
    if not meta["is_valid"] or image is None:
        result.rejection_reasons.append("corrupt_or_unreadable")
        result.check_ms = round((time.perf_counter() - t_start) * 1000, 2)
        return result

    result.width = meta["width"]
    result.height = meta["height"]

    # ------------------------------------------------------------------
    # Check 2: Minimum resolution
    # ------------------------------------------------------------------
    res_ok, res_reason = check_minimum_resolution(image, min_width, min_height)
    if not res_ok:
        result.rejection_reasons.append(res_reason)
        result.check_ms = round((time.perf_counter() - t_start) * 1000, 2)
        return result

    # ------------------------------------------------------------------
    # Check 3: Blur
    # ------------------------------------------------------------------
    result.blur_score = calculate_blur_score(image)
    if result.blur_score < blur_threshold:
        result.rejection_reasons.append("excessive_blur")

    # ------------------------------------------------------------------
    # Check 4: Brightness
    # ------------------------------------------------------------------
    result.brightness = calculate_brightness(image)
    if result.brightness < min_brightness:
        result.rejection_reasons.append("too_dark")
    elif result.brightness > max_brightness:
        result.rejection_reasons.append("overexposed")

    # ------------------------------------------------------------------
    # Check 5: Contrast
    # ------------------------------------------------------------------
    result.contrast = calculate_contrast(image)
    if result.contrast < min_contrast:
        result.rejection_reasons.append("low_contrast")

    # If any quality reasons accumulated → route to MORE_EVIDENCE_REQUIRED
    if result.rejection_reasons:
        result.passed = False
        result.route = "MORE_EVIDENCE_REQUIRED"
        # Still continue to compute hashes so duplicates are recorded
        # but do not overwrite the rejection route with DUPLICATE_REVIEW

    # ------------------------------------------------------------------
    # Check 6: Orientation correction (non-fatal)
    # ------------------------------------------------------------------
    corrected_img, orientation_corrected, orient_warnings = correct_orientation(
        image_path
    )
    result.orientation_corrected = orientation_corrected
    for w in orient_warnings:
        if w not in result.warnings:
            result.warnings.append(w)

    # ------------------------------------------------------------------
    # Check 7: Exact SHA-256 duplicate
    # ------------------------------------------------------------------
    result.sha256 = _compute_sha256(image_path)
    if historical_hashes and result.sha256 in historical_hashes:
        result.is_exact_duplicate = True
        result.duplicate_reference = historical_hashes[result.sha256]
        # Duplicate overrides any prior route (including MORE_EVIDENCE_REQUIRED)
        result.route = "DUPLICATE_REVIEW"
        result.passed = False
        result.check_ms = round((time.perf_counter() - t_start) * 1000, 2)
        return result

    # ------------------------------------------------------------------
    # Check 8: Near-duplicate dHash
    # ------------------------------------------------------------------
    current_dhash = _compute_dhash(image_path)
    if current_dhash and historical_dhashes:
        for hist_hash, ref_path in historical_dhashes.items():
            if _hamming_distance(current_dhash, hist_hash) <= near_dup_threshold:
                result.is_near_duplicate = True
                result.duplicate_reference = ref_path
                result.route = "DUPLICATE_REVIEW"
                result.passed = False
                result.check_ms = round((time.perf_counter() - t_start) * 1000, 2)
                return result

    # ------------------------------------------------------------------
    # Check 9: EXIF summary (informational only — never changes passed/route)
    # ------------------------------------------------------------------
    exif = extract_exif_summary(image_path)
    result.exif_available = exif["exif_available"]
    # exif_absent warning may already be set from correct_orientation()
    if not exif["exif_available"] and "exif_absent" not in result.warnings:
        result.warnings.append("exif_absent")

    # ------------------------------------------------------------------
    # Final routing
    # ------------------------------------------------------------------
    if not result.rejection_reasons:
        result.passed = True
        result.route = "CONTINUE"

    result.check_ms = round((time.perf_counter() - t_start) * 1000, 2)
    return result
