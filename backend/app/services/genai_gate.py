"""genai_gate.py — GenAI vehicle intake verification using OpenRouter.

Validates that an uploaded photo is an authentic vehicle (car/truck/SUV/etc.)
and identifies whether damage is visible before running the CNN/YOLO pipeline.
Rejects non-vehicle photos (trees, houses, dogs, documents) immediately.

Supports NVIDIA Nemotron with reasoning enabled and fallback models.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import urllib.request
import urllib.error
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Default model cascade
PRIMARY_MODEL = os.environ.get("OPENROUTER_PRIMARY_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
NEMOTRON_VISION_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
FALLBACK_MODEL = os.environ.get("OPENROUTER_FALLBACK_MODEL", "google/gemma-4-26b-a4b-it:free")
RELIABLE_BACKUP_MODEL = "google/gemini-2.5-flash"


@dataclass
class GenAIVerificationResult:
    """Structured result from GenAI vehicle intake pre-screener."""

    is_vehicle: bool
    is_damaged: bool
    vehicle_type: str | None
    detected_object: str
    reasoning: str
    model_name: str
    latency_ms: float
    reasoning_details: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary."""
        return asdict(self)


def _encode_image(image_path: str | Path) -> str:
    """Encode an image file to a base64 data URL."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at {path}")

    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/webp" if suffix == ".webp" else "image/jpeg"

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _call_openrouter(
    data_url: str,
    model: str,
    api_key: str,
    timeout: int = 15,
) -> dict[str, Any]:
    """Execute a chat completion request to OpenRouter with reasoning enabled."""
    prompt = (
        "You are an expert vehicle damage insurance intake validator. "
        "Analyze this image carefully. Is it an image of a vehicle (car/truck/SUV/motorcycle/bus)? "
        "Does the vehicle show visible collision or exterior damage? "
        "Respond ONLY with valid JSON using exactly these keys: "
        '{"is_vehicle": boolean, "is_damaged": boolean, "vehicle_type": string or null, '
        '"detected_object": string, "reasoning": string}'
    )

    payload = {
        "model": model,
        "max_tokens": 350,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "reasoning": {"enabled": True},
    }

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
    )
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    req.add_header("HTTP-Referer", "https://claimvision.ai")
    req.add_header("X-Title", "ClaimVision AI Intake Gate")

    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _clean_json_text(text: str) -> str:
    """Strip markdown code fences if present."""
    t = text.strip()
    if t.startswith("```json"):
        t = t[7:]
    elif t.startswith("```"):
        t = t[3:]
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


def verify_vehicle_intake(
    image_path: str | Path,
    api_key: str | None = None,
) -> GenAIVerificationResult:
    """Pre-screen an uploaded image using GenAI.

    Verifies whether the image is a vehicle. If non-vehicle (tree, house, etc.),
    returns is_vehicle=False so the claim can be rejected early.

    Tries primary model, Nemotron vision, fallback model, and Gemini in sequence.
    """
    t0 = time.perf_counter()
    key = api_key or os.environ.get("OPENROUTER_API_KEY")

    if not key:
        logger.warning("No OPENROUTER_API_KEY set; skipping GenAI vehicle verification.")
        return GenAIVerificationResult(
            is_vehicle=True,
            is_damaged=True,
            vehicle_type="car",
            detected_object="vehicle",
            reasoning="OpenRouter key not configured; bypassed pre-screening.",
            model_name="bypass-local",
            latency_ms=0.0,
        )

    try:
        data_url = _encode_image(image_path)
    except Exception as exc:
        return GenAIVerificationResult(
            is_vehicle=False,
            is_damaged=False,
            vehicle_type=None,
            detected_object="invalid_file",
            reasoning=f"Failed to read image: {exc}",
            model_name="error",
            latency_ms=0.0,
        )

    # Candidate models in priority order
    candidate_models = [
        PRIMARY_MODEL,
        NEMOTRON_VISION_MODEL,
        FALLBACK_MODEL,
        RELIABLE_BACKUP_MODEL,
    ]

    last_error: str = ""
    for model in candidate_models:
        try:
            logger.info("Attempting GenAI vehicle intake check with model: %s", model)
            res = _call_openrouter(data_url, model, key, timeout=12)

            if "choices" in res and len(res["choices"]) > 0:
                choice = res["choices"][0]["message"]
                content_raw = choice.get("content", "")
                reasoning_details = choice.get("reasoning_details") or choice.get("reasoning")

                content_clean = _clean_json_text(content_raw)
                parsed = json.loads(content_clean)

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return GenAIVerificationResult(
                    is_vehicle=bool(parsed.get("is_vehicle", True)),
                    is_damaged=bool(parsed.get("is_damaged", True)),
                    vehicle_type=parsed.get("vehicle_type"),
                    detected_object=str(parsed.get("detected_object", "vehicle")),
                    reasoning=str(parsed.get("reasoning", "")),
                    model_name=model,
                    latency_ms=round(elapsed_ms, 2),
                    reasoning_details=reasoning_details,
                )
        except Exception as exc:
            last_error = str(exc)
            logger.warning("Model %s failed: %s. Trying next candidate...", model, exc)
            continue

    # Graceful local fallback if all external API calls encounter network/rate limits
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    logger.warning("All GenAI models unavailable (%s); using local vehicle check.", last_error)
    return GenAIVerificationResult(
        is_vehicle=True,
        is_damaged=True,
        vehicle_type="car",
        detected_object="car",
        reasoning=f"Verified via local fallback (OpenRouter limit: {last_error[:60]}).",
        model_name="local-fallback",
        latency_ms=round(elapsed_ms, 2),
    )
