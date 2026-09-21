"""Phase 0 smoke test — verify the FastAPI health endpoint.

Uses FastAPI's TestClient (via httpx) so no running server is needed.
"""

import os
import sys

from fastapi.testclient import TestClient

# Allow running from repo root: pytest backend/tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.main import app  # noqa: E402

client = TestClient(app)


def test_health_returns_200():
    """GET /api/v1/health must return HTTP 200."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_health_body():
    """Health response must contain status ok and service name."""
    response = client.get("/api/v1/health")
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "claimvision-api"
    assert "version" in body


def test_health_content_type():
    """Health response must be JSON."""
    response = client.get("/api/v1/health")
    assert "application/json" in response.headers["content-type"]
