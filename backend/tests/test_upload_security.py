"""Upload security tests: MIME validation, size limits, and image integrity."""

import io
import os
import sys

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.main import app  # noqa: E402
from backend.app.store import store  # noqa: E402

client = TestClient(app)


def test_upload_invalid_extension():
    """Verify non-image extension is rejected with 422."""
    claim = store.create_claim(policy_number="POL-TEST-001")
    response = client.post(
        f"/api/v1/claims/{claim.id}/images",
        files={"file": ("malicious.exe", b"binary content", "image/jpeg")},
    )
    assert response.status_code == 422
    assert "Unsupported file extension" in response.json()["detail"]


def test_upload_invalid_mime_type():
    """Verify invalid MIME type is rejected with 422."""
    claim = store.create_claim(policy_number="POL-TEST-002")
    response = client.post(
        f"/api/v1/claims/{claim.id}/images",
        files={"file": ("photo.jpg", b"fake image", "application/pdf")},
    )
    assert response.status_code == 422
    assert "Unsupported MIME type" in response.json()["detail"]


def test_upload_empty_file():
    """Verify empty 0-byte file is rejected with 422."""
    claim = store.create_claim(policy_number="POL-TEST-003")
    response = client.post(
        f"/api/v1/claims/{claim.id}/images",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 422
    assert "empty" in response.json()["detail"]


def test_upload_corrupt_image():
    """Verify corrupt non-decodable bytes are rejected with 422."""
    claim = store.create_claim(policy_number="POL-TEST-004")
    response = client.post(
        f"/api/v1/claims/{claim.id}/images",
        files={"file": ("corrupt.jpg", b"corrupt binary bytes that cannot decode", "image/jpeg")},
    )
    assert response.status_code == 422
    assert "corrupt" in response.json()["detail"].lower()


def test_upload_sub_resolution():
    """Verify image below minimum 224x224 resolution is rejected with 422."""
    claim = store.create_claim(policy_number="POL-TEST-005")
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(200, 200, 200)).save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        f"/api/v1/claims/{claim.id}/images",
        files={"file": ("tiny.jpg", buf.read(), "image/jpeg")},
    )
    assert response.status_code == 422
    assert "below minimum requirement" in response.json()["detail"]


def test_upload_valid_image():
    """Verify valid >=224x224 image is accepted and attached to claim."""
    claim = store.create_claim(policy_number="POL-TEST-006")
    buf = io.BytesIO()
    Image.new("RGB", (300, 300), color=(150, 150, 150)).save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        f"/api/v1/claims/{claim.id}/images",
        files={"file": ("valid.jpg", buf.read(), "image/jpeg")},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["claim_id"] == claim.id
    assert data["width"] == 300
    assert data["height"] == 300
    assert len(data["sha256"]) == 64
