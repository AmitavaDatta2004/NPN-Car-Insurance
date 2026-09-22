"""store.py — SQLite-backed persistent store for ClaimVision AI.

Task ID : INT-001 / BE-001
Phase   : 13 & 16 (Backend Database & Reviewer Workspace)

Provides persistent SQLite storage (backend/claimvision.db) for claims, images,
assessments, reviews, and status event timelines via SQLAlchemy 2.0.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from sqlalchemy.orm import Session

from backend.app.db.models import (
    AssessmentDB,
    ClaimDB,
    ImageDB,
    ReviewDB,
    TimelineDB,
)
from backend.app.db.session import Base, SessionLocal, engine


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
    """Record of an uploaded evidence image."""

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
    """Record of an insurance claim."""

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


def _claim_db_to_record(db_claim: ClaimDB) -> ClaimRecord:
    """Convert a ClaimDB SQLAlchemy model to a ClaimRecord dataclass."""
    images = [
        ImageRecord(
            id=img.id,
            claim_id=img.claim_id,
            filename=img.filename,
            local_path=img.local_path,
            file_size_bytes=img.file_size_bytes,
            mime_type=img.mime_type,
            sha256=img.sha256,
            width=img.width,
            height=img.height,
            created_at=img.created_at,
        )
        for img in db_claim.images
    ]

    timeline = [
        TimelineEvent(
            id=evt.id,
            claim_id=evt.claim_id,
            from_status=evt.from_status,
            to_status=evt.to_status,
            reason=evt.reason,
            actor=evt.actor,
            timestamp=evt.timestamp,
        )
        for evt in db_claim.timeline
    ]

    assessment = db_claim.assessment.to_dict() if db_claim.assessment else None
    review = db_claim.review.to_dict() if db_claim.review else None

    return ClaimRecord(
        id=db_claim.id,
        policy_number=db_claim.policy_number,
        vehicle_make=db_claim.vehicle_make,
        vehicle_model=db_claim.vehicle_model,
        vehicle_year=db_claim.vehicle_year,
        vehicle_segment=db_claim.vehicle_segment,
        incident_description=db_claim.incident_description,
        incident_date=db_claim.incident_date,
        status=ClaimStatus(db_claim.status),
        images=images,
        assessment=assessment,
        review=review,
        timeline=timeline,
        created_at=db_claim.created_at,
        updated_at=db_claim.updated_at,
    )


class DatabaseClaimStore:
    """Persistent SQLite store for ClaimVision AI."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # Initialize tables on startup
        Base.metadata.create_all(bind=engine)

    def _get_session(self) -> Session:
        return SessionLocal()

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
        """Create a new draft claim in SQLite."""
        with self._lock:
            cid = str(uuid.uuid4())
            now = utc_now_iso()
            session = self._get_session()
            try:
                db_claim = ClaimDB(
                    id=cid,
                    policy_number=policy_number,
                    vehicle_make=vehicle_make,
                    vehicle_model=vehicle_model,
                    vehicle_year=vehicle_year,
                    vehicle_segment=vehicle_segment,
                    incident_description=incident_description,
                    incident_date=incident_date or now[:10],
                    status=ClaimStatus.DRAFT.value,
                    created_at=now,
                    updated_at=now,
                )
                init_event = TimelineDB(
                    id=str(uuid.uuid4()),
                    claim_id=cid,
                    from_status="",
                    to_status=ClaimStatus.DRAFT.value,
                    reason="Claim draft initiated",
                    actor="policyholder",
                    timestamp=now,
                )
                db_claim.timeline.append(init_event)
                session.add(db_claim)
                session.commit()
                session.refresh(db_claim)
                return _claim_db_to_record(db_claim)
            finally:
                session.close()

    def get_claim(self, claim_id: str) -> ClaimRecord | None:
        """Fetch claim by ID from SQLite."""
        with self._lock:
            session = self._get_session()
            try:
                db_claim = session.query(ClaimDB).filter(ClaimDB.id == claim_id).first()
                if not db_claim:
                    return None
                return _claim_db_to_record(db_claim)
            finally:
                session.close()

    def list_claims(self) -> list[ClaimRecord]:
        """List all claims ordered by creation time descending."""
        with self._lock:
            session = self._get_session()
            try:
                claims = session.query(ClaimDB).order_by(ClaimDB.created_at.desc()).all()
                return [_claim_db_to_record(c) for c in claims]
            finally:
                session.close()

    def update_claim(
        self,
        claim_id: str,
        **fields: Any,
    ) -> ClaimRecord | None:
        """Update draft claim fields in SQLite."""
        with self._lock:
            session = self._get_session()
            try:
                db_claim = session.query(ClaimDB).filter(ClaimDB.id == claim_id).first()
                if not db_claim:
                    return None

                for k, v in fields.items():
                    if k == "review" and isinstance(v, dict):
                        # Update or create review record
                        if db_claim.review:
                            db_claim.review.reviewer_id = v.get("reviewer_id", db_claim.review.reviewer_id)
                            db_claim.review.decision = v.get("decision", db_claim.review.decision)
                            db_claim.review.notes = v.get("notes", db_claim.review.notes)
                            db_claim.review.overrides_json = json.dumps(v.get("overrides", {}))
                            db_claim.review.timestamp = v.get("timestamp", utc_now_iso())
                        else:
                            new_review = ReviewDB(
                                id=str(uuid.uuid4()),
                                claim_id=claim_id,
                                reviewer_id=v.get("reviewer_id", "adjuster_1"),
                                decision=v.get("decision", "PENDING"),
                                notes=v.get("notes", ""),
                                overrides_json=json.dumps(v.get("overrides", {})),
                                timestamp=v.get("timestamp", utc_now_iso()),
                            )
                            db_claim.review = new_review
                    elif hasattr(db_claim, k) and v is not None:
                        setattr(db_claim, k, v)

                db_claim.updated_at = utc_now_iso()
                session.commit()
                session.refresh(db_claim)
                return _claim_db_to_record(db_claim)
            finally:
                session.close()

    def transition_status(
        self,
        claim_id: str,
        to_status: ClaimStatus,
        reason: str = "",
        actor: str = "system",
    ) -> ClaimRecord | None:
        """Transition claim state with validation in SQLite."""
        with self._lock:
            session = self._get_session()
            try:
                db_claim = session.query(ClaimDB).filter(ClaimDB.id == claim_id).first()
                if not db_claim:
                    return None

                from_status = ClaimStatus(db_claim.status)
                valid_targets = ALLOWED_TRANSITIONS.get(from_status, set())
                if to_status not in valid_targets and to_status != from_status:
                    raise ValueError(
                        f"Invalid transition from {from_status.value} to {to_status.value}"
                    )

                now = utc_now_iso()
                db_claim.status = to_status.value
                db_claim.updated_at = now

                ev = TimelineDB(
                    id=str(uuid.uuid4()),
                    claim_id=claim_id,
                    from_status=from_status.value,
                    to_status=to_status.value,
                    reason=reason or f"Transitioned to {to_status.value}",
                    actor=actor,
                    timestamp=now,
                )
                db_claim.timeline.append(ev)
                session.commit()
                session.refresh(db_claim)
                return _claim_db_to_record(db_claim)
            finally:
                session.close()

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
        """Attach an uploaded image to a claim in SQLite."""
        with self._lock:
            session = self._get_session()
            try:
                db_claim = session.query(ClaimDB).filter(ClaimDB.id == claim_id).first()
                if not db_claim:
                    return None

                img_id = str(uuid.uuid4())
                now = utc_now_iso()
                db_image = ImageDB(
                    id=img_id,
                    claim_id=claim_id,
                    filename=filename,
                    local_path=local_path,
                    file_size_bytes=file_size_bytes,
                    mime_type=mime_type,
                    sha256=sha256,
                    width=width,
                    height=height,
                    created_at=now,
                )
                db_claim.images.append(db_image)
                db_claim.updated_at = now
                session.commit()
                return ImageRecord(
                    id=img_id,
                    claim_id=claim_id,
                    filename=filename,
                    local_path=local_path,
                    file_size_bytes=file_size_bytes,
                    mime_type=mime_type,
                    sha256=sha256,
                    width=width,
                    height=height,
                    created_at=now,
                )
            finally:
                session.close()

    def get_image(self, image_id: str) -> ImageRecord | None:
        """Fetch image by ID from SQLite."""
        with self._lock:
            session = self._get_session()
            try:
                img = session.query(ImageDB).filter(ImageDB.id == image_id).first()
                if not img:
                    return None
                return ImageRecord(
                    id=img.id,
                    claim_id=img.claim_id,
                    filename=img.filename,
                    local_path=img.local_path,
                    file_size_bytes=img.file_size_bytes,
                    mime_type=img.mime_type,
                    sha256=img.sha256,
                    width=img.width,
                    height=img.height,
                    created_at=img.created_at,
                )
            finally:
                session.close()

    def delete_image(self, image_id: str) -> bool:
        """Remove an image from SQLite."""
        with self._lock:
            session = self._get_session()
            try:
                img = session.query(ImageDB).filter(ImageDB.id == image_id).first()
                if not img:
                    return False
                session.delete(img)
                session.commit()
                return True
            finally:
                session.close()

    def save_assessment(
        self,
        claim_id: str,
        assessment_data: dict[str, Any],
    ) -> ClaimRecord | None:
        """Attach completed assessment and transition status based on route in SQLite."""
        with self._lock:
            session = self._get_session()
            try:
                db_claim = session.query(ClaimDB).filter(ClaimDB.id == claim_id).first()
                if not db_claim:
                    return None

                route = assessment_data.get("route", "TECHNICAL_REVIEW")
                route_map = {
                    "FAST_TRACK_ELIGIBLE": ClaimStatus.FAST_TRACK_ELIGIBLE,
                    "FRAUD_REVIEW": ClaimStatus.FRAUD_REVIEW,
                    "MANUAL_DAMAGE_REVIEW": ClaimStatus.MANUAL_DAMAGE_REVIEW,
                    "MORE_EVIDENCE_REQUIRED": ClaimStatus.MORE_EVIDENCE_REQUIRED,
                    "TECHNICAL_REVIEW": ClaimStatus.TECHNICAL_REVIEW,
                }
                target_status = route_map.get(route, ClaimStatus.TECHNICAL_REVIEW)

                now = utc_now_iso()
                from_status = db_claim.status

                # If assessment already exists, update it
                if db_claim.assessment:
                    db_claim.assessment.image_path = assessment_data.get("image_path", "")
                    db_claim.assessment.route = route
                    db_claim.assessment.reason_codes_json = json.dumps(assessment_data.get("reason_codes", []))
                    db_claim.assessment.genai_gate_json = json.dumps(assessment_data.get("genai_gate"))
                    db_claim.assessment.quality_json = json.dumps(assessment_data.get("quality"))
                    db_claim.assessment.fraud_json = json.dumps(assessment_data.get("fraud"))
                    db_claim.assessment.severity_json = json.dumps(assessment_data.get("severity"))
                    db_claim.assessment.location_json = json.dumps(assessment_data.get("location"))
                    db_claim.assessment.detections_json = json.dumps(assessment_data.get("detections", []))
                    db_claim.assessment.cost_json = json.dumps(assessment_data.get("cost"))
                    db_claim.assessment.model_versions_json = json.dumps(assessment_data.get("model_versions", {}))
                    db_claim.assessment.inference_ms = assessment_data.get("inference_ms", 0)
                    db_claim.assessment.warnings_json = json.dumps(assessment_data.get("warnings", []))
                else:
                    db_assessment = AssessmentDB(
                        id=str(uuid.uuid4()),
                        claim_id=claim_id,
                        image_path=assessment_data.get("image_path", ""),
                        route=route,
                        reason_codes_json=json.dumps(assessment_data.get("reason_codes", [])),
                        genai_gate_json=json.dumps(assessment_data.get("genai_gate")),
                        quality_json=json.dumps(assessment_data.get("quality")),
                        fraud_json=json.dumps(assessment_data.get("fraud")),
                        severity_json=json.dumps(assessment_data.get("severity")),
                        location_json=json.dumps(assessment_data.get("location")),
                        detections_json=json.dumps(assessment_data.get("detections", [])),
                        cost_json=json.dumps(assessment_data.get("cost")),
                        model_versions_json=json.dumps(assessment_data.get("model_versions", {})),
                        inference_ms=assessment_data.get("inference_ms", 0),
                        warnings_json=json.dumps(assessment_data.get("warnings", [])),
                        created_at=now,
                    )
                    db_claim.assessment = db_assessment

                db_claim.status = target_status.value
                db_claim.updated_at = now

                ev = TimelineDB(
                    id=str(uuid.uuid4()),
                    claim_id=claim_id,
                    from_status=from_status,
                    to_status=target_status.value,
                    reason=f"Assessment complete: route={route}",
                    actor="assessment_engine",
                    timestamp=now,
                )
                db_claim.timeline.append(ev)
                session.commit()
                session.refresh(db_claim)
                return _claim_db_to_record(db_claim)
            finally:
                session.close()

    def save_review(
        self,
        claim_id: str,
        reviewer_id: str,
        decision: str,
        notes: str,
        overrides: dict[str, Any] | None = None,
    ) -> ClaimRecord | None:
        """Save human adjuster review decision without overwriting AI findings in SQLite."""
        with self._lock:
            session = self._get_session()
            try:
                db_claim = session.query(ClaimDB).filter(ClaimDB.id == claim_id).first()
                if not db_claim:
                    return None

                now = utc_now_iso()
                if db_claim.review:
                    db_claim.review.reviewer_id = reviewer_id
                    db_claim.review.decision = decision
                    db_claim.review.notes = notes
                    db_claim.review.overrides_json = json.dumps(overrides or {})
                    db_claim.review.timestamp = now
                else:
                    new_review = ReviewDB(
                        id=str(uuid.uuid4()),
                        claim_id=claim_id,
                        reviewer_id=reviewer_id,
                        decision=decision,
                        notes=notes,
                        overrides_json=json.dumps(overrides or {}),
                        timestamp=now,
                    )
                    db_claim.review = new_review

                from_status = db_claim.status
                db_claim.status = ClaimStatus.COMPLETED.value
                db_claim.updated_at = now

                ev = TimelineDB(
                    id=str(uuid.uuid4()),
                    claim_id=claim_id,
                    from_status=from_status,
                    to_status=ClaimStatus.COMPLETED.value,
                    reason=f"Review decision: {decision} by {reviewer_id}",
                    actor=reviewer_id,
                    timestamp=now,
                )
                db_claim.timeline.append(ev)
                session.commit()
                session.refresh(db_claim)
                return _claim_db_to_record(db_claim)
            finally:
                session.close()

    def get_review_queue(self) -> list[ClaimRecord]:
        """Fetch claims currently pending human adjuster review from SQLite."""
        with self._lock:
            session = self._get_session()
            try:
                review_statuses = [
                    ClaimStatus.FRAUD_REVIEW.value,
                    ClaimStatus.MANUAL_DAMAGE_REVIEW.value,
                    ClaimStatus.TECHNICAL_REVIEW.value,
                ]
                claims = (
                    session.query(ClaimDB)
                    .filter(
                        ClaimDB.status.in_(review_statuses),
                        ClaimDB.review == None,  # noqa: E711
                    )
                    .order_by(ClaimDB.created_at.desc())
                    .all()
                )
                return [_claim_db_to_record(c) for c in claims]
            finally:
                session.close()

    def get_dashboard_summary(self) -> dict[str, Any]:
        """Compute aggregated KPIs from SQLite."""
        with self._lock:
            session = self._get_session()
            try:
                all_claims_db = session.query(ClaimDB).all()
                all_claims = [_claim_db_to_record(c) for c in all_claims_db]
                total = len(all_claims)

                counts = {s.value: 0 for s in ClaimStatus}
                for c in all_claims:
                    counts[c.status.value] += 1

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
            finally:
                session.close()


# Persistent SQLite singleton instance
store = DatabaseClaimStore()
