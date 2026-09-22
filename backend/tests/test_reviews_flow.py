"""Integration tests for reviewer queue, corrections, decisions, and dashboard summary."""

import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.main import app  # noqa: E402
from backend.app.store import ClaimStatus, store  # noqa: E402

client = TestClient(app)


def test_review_queue_and_correction_flow():
    """Verify reviewer queue filtering, saving corrections, and decision submission."""
    # 1. Seed a claim in MANUAL_DAMAGE_REVIEW
    claim = store.create_claim(policy_number="POL-REV-001")
    store.save_assessment(
        claim.id,
        {
            "route": "MANUAL_DAMAGE_REVIEW",
            "reason_codes": ["severe_damage"],
            "severity": {"predicted_class": "severe", "confidence": 0.88},
            "location": {"predicted_part": "front_bumper", "confidence": 0.82},
            "cost": {"min_cost": 25000, "max_cost": 65000},
            "fraud": {"probability": 0.08},
        },
    )

    # 2. Check it appears in review queue
    q_resp = client.get("/api/v1/reviews/queue")
    assert q_resp.status_code == 200
    queue = q_resp.json()
    matching = [c for c in queue if c["id"] == claim.id]
    assert len(matching) == 1
    assert matching[0]["status"] == "MANUAL_DAMAGE_REVIEW"

    # 3. Save reviewer correction
    corr_resp = client.patch(
        f"/api/v1/reviews/{claim.id}",
        json={
            "reviewer_id": "adjuster_alice",
            "notes": "Verified bumper crack physically; repair feasible.",
            "overrides": {"action": "repair", "approved_amount": 35000},
        },
    )
    assert corr_resp.status_code == 200
    rev_claim = corr_resp.json()
    assert rev_claim["review"]["reviewer_id"] == "adjuster_alice"
    assert rev_claim["review"]["overrides"]["approved_amount"] == 35000
    # Original AI assessment must NOT be overwritten!
    assert rev_claim["assessment"]["severity"]["predicted_class"] == "severe"

    # 4. Submit final decision
    dec_resp = client.post(
        f"/api/v1/reviews/{claim.id}/decision",
        json={
            "reviewer_id": "adjuster_alice",
            "decision": "APPROVED",
            "notes": "Approved with adjusted estimate.",
            "overrides": {"approved_amount": 35000},
        },
    )
    assert dec_resp.status_code == 200
    final_claim = dec_resp.json()
    assert final_claim["status"] == "COMPLETED"
    assert final_claim["review"]["decision"] == "APPROVED"

    # 5. Verify claim no longer in review queue
    q_resp_after = client.get("/api/v1/reviews/queue")
    assert not any(c["id"] == claim.id for c in q_resp_after.json())


def test_dashboard_summary_kpis():
    """Verify GET /api/v1/dashboard/summary computes aggregate statistics."""
    resp = client.get("/api/v1/dashboard/summary")
    assert resp.status_code == 200
    data = resp.json()

    assert "total_claims" in data
    assert "status_counts" in data
    assert "fast_track_count" in data
    assert "average_cost" in data
    assert "severity_distribution" in data
    assert "location_distribution" in data
    assert "override_rate" in data
    assert isinstance(data["total_claims"], int)
    assert data["total_claims"] >= 1
