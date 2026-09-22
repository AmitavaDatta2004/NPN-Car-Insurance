"""Integration tests for end-to-end claim lifecycle: create, upload, submit, assess, timeline."""

import io
import os
import sys

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.main import app  # noqa: E402
from backend.app.store import store  # noqa: E402

client = TestClient(app)


def _make_valid_image_bytes() -> bytes:
    """Generate high-contrast JPEG bytes (>224x224) that pass quality checks."""
    arr = np.zeros((300, 300, 3), dtype=np.uint8)
    for i in range(300):
        for j in range(300):
            arr[i, j] = [(i * 7) % 256, (j * 11) % 256, ((i + j) * 5) % 256]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return buf.read()


def test_claim_lifecycle_create_and_read():
    """Verify claim creation and retrieval."""
    resp = client.post(
        "/api/v1/claims",
        json={
            "policy_number": "POL-LIFE-001",
            "vehicle_make": "Honda",
            "vehicle_model": "Civic",
            "vehicle_year": 2021,
            "vehicle_segment": "sedan",
            "incident_description": "Hit parked car in parking lot",
        },
    )
    assert resp.status_code == 201
    claim = resp.json()
    assert claim["status"] == "DRAFT"
    assert claim["policy_number"] == "POL-LIFE-001"

    # Fetch by ID
    get_resp = client.get(f"/api/v1/claims/{claim['id']}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == claim["id"]


def test_claim_cannot_submit_without_images():
    """Verify submit fails if no images uploaded."""
    resp = client.post("/api/v1/claims", json={"policy_number": "POL-EMPTY-001"})
    claim_id = resp.json()["id"]

    sub_resp = client.post(f"/api/v1/claims/{claim_id}/submit")
    assert sub_resp.status_code == 422
    assert "zero evidence images" in sub_resp.json()["detail"]


def test_claim_full_assessment_flow():
    """Verify create -> upload image -> submit -> assess -> read assessment -> timeline."""
    # 1. Create claim
    c_resp = client.post(
        "/api/v1/claims",
        json={
            "policy_number": "POL-ASSESS-001",
            "vehicle_make": "Hyundai",
            "vehicle_model": "i20",
            "vehicle_year": 2023,
            "vehicle_segment": "compact",
        },
    )
    claim_id = c_resp.json()["id"]

    # 2. Upload image
    img_bytes = _make_valid_image_bytes()
    up_resp = client.post(
        f"/api/v1/claims/{claim_id}/images",
        files={"file": ("front_bumper.jpg", img_bytes, "image/jpeg")},
    )
    assert up_resp.status_code == 201
    image_id = up_resp.json()["id"]

    # 3. Submit claim
    sub_resp = client.post(f"/api/v1/claims/{claim_id}/submit")
    assert sub_resp.status_code == 200
    assert sub_resp.json()["status"] == "SUBMITTED"

    # 4. Trigger assessment
    assess_resp = client.post(f"/api/v1/claims/{claim_id}/assess")
    assert assess_resp.status_code == 200
    assessment = assess_resp.json()
    assert "route" in assessment
    assert "severity" in assessment
    assert "cost" in assessment
    assert "location" in assessment
    assert assessment["cost"]["vehicle_segment"] == "compact"

    # 5. Check status polling
    poll_resp = client.get(f"/api/v1/assessments/{claim_id}/status")
    assert poll_resp.status_code == 200
    status_data = poll_resp.json()
    assert status_data["is_complete"] is True
    assert status_data["has_assessment"] is True

    # 6. Check timeline has audit entries
    time_resp = client.get(f"/api/v1/claims/{claim_id}/timeline")
    assert time_resp.status_code == 200
    events = time_resp.json()
    assert len(events) >= 3  # DRAFT -> SUBMITTED -> ASSESSING -> (ROUTE)
