"""store.py — Thread-safe in-memory store for ClaimVision AI demo.

Task ID : BE-001
Phase   : 13 (Backend Foundation & Assessment APIs)
Owner   : Member 5 (Backend)

Provides fast, zero-database in-memory storage for claims, images, assessments,
reviews, and status event timelines.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class ClaimStatus(str, Enum):
    """Claim state machine enumeration matching README §16.3."""

    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    VALIDATING = "VALIDATING"
    MORE_EVIDENCE_REQUIRED = "MORE_EVIDENCE_REQUIRED"
    ASSESSING_FRAUD = "ASSESSING_FRAUD"
    FRAUD_REVIEW = "FRAUD_REVIEW"
    ASSESSING_DAMAGE = "ASSESSING_DAMAGE"
    MANUAL_DAMAGE_REVIEW = "MANUAL_DAMAGE_REVIEW"
    FAST_TRACK_ELIGIBLE = "FAST_TRACK_ELIGIBLE"
    TECHNICAL_REVIEW = "TECHNICAL_REVIEW"
    COMPLETED = "COMPLETED"


# Allowed state transitions for validation
ALLOWED_TRANSITIONS: dict[ClaimStatus, set[ClaimStatus]] = {
    ClaimStatus.DRAFT: {ClaimStatus.SUBMITTED},
    ClaimStatus.SUBMITTED: {
        ClaimStatus.VALIDATING,
        ClaimStatus.ASSESSING_FRAUD,
        ClaimStatus.MORE_EVIDENCE_REQUIRED,
    },
    ClaimStatus.VALIDATING: {
        ClaimStatus.ASSESSING_FRAUD,
        ClaimStatus.MORE_EVIDENCE_REQUIRED,
    },
    ClaimStatus.ASSESSING_FRAUD: {
        ClaimStatus.FRAUD_REVIEW,
        ClaimStatus.ASSESSING_DAMAGE,
        ClaimStatus.TECHNICAL_REVIEW,
    },
    ClaimStatus.ASSESSING_DAMAGE: {
        ClaimStatus.MANUAL_DAMAGE_REVIEW,
        ClaimStatus.FAST_TRACK_ELIGIBLE,
        ClaimStatus.TECHNICAL_REVIEW,
    },
    ClaimStatus.FRAUD_REVIEW: {ClaimStatus.COMPLETED, ClaimStatus.MORE_EVIDENCE_REQUIRED},
    ClaimStatus.MANUAL_DAMAGE_REVIEW: {ClaimStatus.COMPLETED, ClaimStatus.MORE_EVIDENCE_REQUIRED},
    ClaimStatus.FAST_TRACK_ELIGIBLE: {ClaimStatus.COMPLETED},
    ClaimStatus.MORE_EVIDENCE_REQUIRED: {ClaimStatus.SUBMITTED},
    ClaimStatus.TECHNICAL_REVIEW: {ClaimStatus.COMPLETED, ClaimStatus.ASSESSING_FRAUD},
    ClaimStatus.COMPLETED: set(),
}


@dataclass
class ImageRecord:
    """In-memory record of an uploaded evidence image."""

    id: str
    claim_id: str
    filename: str
    local_path: str
    file_size_bytes: int
    mime_type: str
    sha256: str
    width: int
    height: int
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TimelineEvent:
    """Audit log entry of a claim state transition."""

    id: str
    claim_id: str
    from_status: str
    to_status: str
    reason: str
    actor: str
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimRecord:
    """In-memory record of an insurance claim."""

    id: str
    policy_number: str
    vehicle_make: str
    vehicle_model: str
    vehicle_year: int
    vehicle_segment: str
    incident_description: str
    incident_date: str
    status: ClaimStatus = ClaimStatus.DRAFT
    images: list[ImageRecord] = field(default_factory=list)
    assessment: dict[str, Any] | None = None
    review: dict[str, Any] | None = None
    timeline: list[TimelineEvent] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "policy_number": self.policy_number,
            "vehicle_make": self.vehicle_make,
            "vehicle_model": self.vehicle_model,
            "vehicle_year": self.vehicle_year,
            "vehicle_segment": self.vehicle_segment,
            "incident_description": self.incident_description,
            "incident_date": self.incident_date,
            "status": self.status.value,
            "images": [img.to_dict() for img in self.images],
            "assessment": self.assessment,
            "review": self.review,
            "timeline": [ev.to_dict() for ev in self.timeline],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class InMemoryStore:
    """Thread-safe store for ClaimVision AI demo data."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._claims: dict[str, ClaimRecord] = {}
        self._images: dict[str, ImageRecord] = {}

    def create_claim(
        self,
        policy_number: str,
        vehicle_make: str = "Unknown",
        vehicle_model: str = "Unknown",
        vehicle_year: int = 2022,
        vehicle_segment: str = "compact",
        incident_description: str = "",
        incident_date: str = "",
    ) -> ClaimRecord:
        """Create a new draft claim."""
        with self._lock:
            cid = str(uuid.uuid4())
            now = utc_now_iso()
            claim = ClaimRecord(
                id=cid,
                policy_number=policy_number,
                vehicle_make=vehicle_make,
                vehicle_model=vehicle_model,
                vehicle_year=vehicle_year,
                vehicle_segment=vehicle_segment,
                incident_description=incident_description,
                incident_date=incident_date or now[:10],
                status=ClaimStatus.DRAFT,
                created_at=now,
                updated_at=now,
            )
            # Record initial creation event
            init_event = TimelineEvent(
                id=str(uuid.uuid4()),
                claim_id=cid,
                from_status="",
                to_status=ClaimStatus.DRAFT.value,
                reason="Claim draft initiated",
                actor="policyholder",
                timestamp=now,
            )
            claim.timeline.append(init_event)
            self._claims[cid] = claim
            return claim

    def get_claim(self, claim_id: str) -> ClaimRecord | None:
        """Fetch claim by ID."""
        with self._lock:
            return self._claims.get(claim_id)

    def list_claims(self) -> list[ClaimRecord]:
        """List all claims ordered by creation time descending."""
        with self._lock:
            return sorted(
                self._claims.values(),
                key=lambda c: c.created_at,
                reverse=True,
            )

    def update_claim(
        self,
        claim_id: str,
        **fields: Any,
    ) -> ClaimRecord | None:
        """Update draft claim fields."""
        with self._lock:
            claim = self._claims.get(claim_id)
            if not claim:
                return None
            for k, v in fields.items():
                if hasattr(claim, k) and v is not None:
                    setattr(claim, k, v)
            claim.updated_at = utc_now_iso()
            return claim

    def transition_status(
        self,
        claim_id: str,
        to_status: ClaimStatus,
        reason: str = "",
        actor: str = "system",
    ) -> ClaimRecord | None:
        """Transition claim state with validation."""
        with self._lock:
            claim = self._claims.get(claim_id)
            if not claim:
                return None

            from_status = claim.status
            # Validate transition if not forcing
            valid_targets = ALLOWED_TRANSITIONS.get(from_status, set())
            if to_status not in valid_targets and to_status != from_status:
                raise ValueError(
                    f"Invalid transition from {from_status.value} to {to_status.value}"
                )

            claim.status = to_status
            claim.updated_at = utc_now_iso()
            ev = TimelineEvent(
                id=str(uuid.uuid4()),
                claim_id=claim_id,
                from_status=from_status.value,
                to_status=to_status.value,
                reason=reason or f"Transitioned to {to_status.value}",
                actor=actor,
                timestamp=claim.updated_at,
            )
            claim.timeline.append(ev)
            return claim

    def add_image(
        self,
        claim_id: str,
        filename: str,
        local_path: str,
        file_size_bytes: int,
        mime_type: str,
        sha256: str,
        width: int,
        height: int,
    ) -> ImageRecord | None:
        """Attach an uploaded image to a claim."""
        with self._lock:
            claim = self._claims.get(claim_id)
            if not claim:
                return None

            img_id = str(uuid.uuid4())
            record = ImageRecord(
                id=img_id,
                claim_id=claim_id,
                filename=filename,
                local_path=local_path,
                file_size_bytes=file_size_bytes,
                mime_type=mime_type,
                sha256=sha256,
                width=width,
                height=height,
            )
            self._images[img_id] = record
            claim.images.append(record)
            claim.updated_at = utc_now_iso()
            return record

    def get_image(self, image_id: str) -> ImageRecord | None:
        """Fetch image by ID."""
        with self._lock:
            return self._images.get(image_id)

    def delete_image(self, image_id: str) -> bool:
        """Remove a draft image from its claim."""
        with self._lock:
            img = self._images.pop(image_id, None)
            if not img:
                return False
            claim = self._claims.get(img.claim_id)
            if claim:
                claim.images = [i for i in claim.images if i.id != image_id]
                claim.updated_at = utc_now_iso()
            return True

    def save_assessment(
        self,
        claim_id: str,
        assessment_data: dict[str, Any],
    ) -> ClaimRecord | None:
        """Attach completed assessment and transition status based on route."""
        with self._lock:
            claim = self._claims.get(claim_id)
            if not claim:
                return None

            claim.assessment = assessment_data
            route = assessment_data.get("route", "TECHNICAL_REVIEW")

            # Map route to terminal claim status
            route_map = {
                "FAST_TRACK_ELIGIBLE": ClaimStatus.FAST_TRACK_ELIGIBLE,
                "FRAUD_REVIEW": ClaimStatus.FRAUD_REVIEW,
                "MANUAL_DAMAGE_REVIEW": ClaimStatus.MANUAL_DAMAGE_REVIEW,
                "MORE_EVIDENCE_REQUIRED": ClaimStatus.MORE_EVIDENCE_REQUIRED,
                "TECHNICAL_REVIEW": ClaimStatus.TECHNICAL_REVIEW,
            }
            target_status = route_map.get(route, ClaimStatus.TECHNICAL_REVIEW)

            # Bypass standard transition chain for assessment outcome
            from_status = claim.status
            claim.status = target_status
            claim.updated_at = utc_now_iso()
            ev = TimelineEvent(
                id=str(uuid.uuid4()),
                claim_id=claim_id,
                from_status=from_status.value,
                to_status=target_status.value,
                reason=f"Assessment complete: route={route}",
                actor="assessment_engine",
                timestamp=claim.updated_at,
            )
            claim.timeline.append(ev)
            return claim

    def save_review(
        self,
        claim_id: str,
        reviewer_id: str,
        decision: str,
        notes: str,
        overrides: dict[str, Any] | None = None,
    ) -> ClaimRecord | None:
        """Save human adjuster review decision without overwriting AI findings."""
        with self._lock:
            claim = self._claims.get(claim_id)
            if not claim:
                return None

            now = utc_now_iso()
            claim.review = {
                "reviewer_id": reviewer_id,
                "decision": decision,
                "notes": notes,
                "overrides": overrides or {},
                "timestamp": now,
            }
            from_status = claim.status
            claim.status = ClaimStatus.COMPLETED
            claim.updated_at = now
            ev = TimelineEvent(
                id=str(uuid.uuid4()),
                claim_id=claim_id,
                from_status=from_status.value,
                to_status=ClaimStatus.COMPLETED.value,
                reason=f"Review decision: {decision} by {reviewer_id}",
                actor=reviewer_id,
                timestamp=now,
            )
            claim.timeline.append(ev)
            return claim

    def get_review_queue(self) -> list[ClaimRecord]:
        """Fetch claims currently pending human adjuster review."""
        with self._lock:
            review_statuses = {
                ClaimStatus.FRAUD_REVIEW,
                ClaimStatus.MANUAL_DAMAGE_REVIEW,
                ClaimStatus.TECHNICAL_REVIEW,
            }
            return [
                c for c in self._claims.values()
                if c.status in review_statuses and c.review is None
            ]

    def get_dashboard_summary(self) -> dict[str, Any]:
        """Compute aggregated KPIs for the reviewer dashboard."""
        with self._lock:
            all_claims = list(self._claims.values())
            total = len(all_claims)

            counts = {s.value: 0 for s in ClaimStatus}
            for c in all_claims:
                counts[c.status.value] += 1

            # Severity distribution from assessments
            severity_dist: dict[str, int] = {"minor": 0, "moderate": 0, "severe": 0}
            location_dist: dict[str, int] = {}
            total_cost_min = 0.0
            total_cost_max = 0.0
            cost_count = 0
            fraud_scores: list[float] = []

            for c in all_claims:
                if c.assessment:
                    sev = c.assessment.get("severity", {})
                    if sev and "predicted_class" in sev:
                        cls = sev["predicted_class"]
                        severity_dist[cls] = severity_dist.get(cls, 0) + 1

                    loc = c.assessment.get("location", {})
                    if loc and "predicted_part" in loc:
                        part = loc["predicted_part"]
                        location_dist[part] = location_dist.get(part, 0) + 1

                    cost = c.assessment.get("cost", {})
                    if cost and "min_cost" in cost:
                        total_cost_min += cost.get("min_cost", 0.0)
                        total_cost_max += cost.get("max_cost", 0.0)
                        cost_count += 1

                    fraud = c.assessment.get("fraud", {})
                    if fraud and "probability" in fraud:
                        fraud_scores.append(fraud["probability"])

            avg_cost_min = (total_cost_min / cost_count) if cost_count > 0 else 0.0
            avg_cost_max = (total_cost_max / cost_count) if cost_count > 0 else 0.0

            # Override rate: claims with reviews that modified AI outcome
            reviewed_count = sum(1 for c in all_claims if c.review is not None)
            overrides_count = sum(
                1 for c in all_claims
                if c.review and bool(c.review.get("overrides"))
            )
            override_rate = (overrides_count / reviewed_count) if reviewed_count > 0 else 0.0

            return {
                "total_claims": total,
                "status_counts": counts,
                "fast_track_count": counts[ClaimStatus.FAST_TRACK_ELIGIBLE.value],
                "fraud_review_count": counts[ClaimStatus.FRAUD_REVIEW.value],
                "damage_review_count": counts[ClaimStatus.MANUAL_DAMAGE_REVIEW.value],
                "completed_count": counts[ClaimStatus.COMPLETED.value],
                "average_cost": {
                    "min": round(avg_cost_min, 2),
                    "max": round(avg_cost_max, 2),
                },
                "severity_distribution": severity_dist,
                "location_distribution": location_dist,
                "fraud_score_distribution": fraud_scores,
                "override_rate": round(override_rate, 4),
            }


# Global in-memory singleton
store = InMemoryStore()
