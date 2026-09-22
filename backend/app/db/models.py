"""models.py — SQLAlchemy ORM models for claims, images, assessments, reviews, and timelines.

Task ID : INT-001
"""

import json
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from backend.app.db.session import Base


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ClaimDB(Base):
    __tablename__ = "claims"

    id = Column(String(64), primary_key=True, index=True)
    policy_number = Column(String(64), index=True, nullable=False)
    vehicle_make = Column(String(64), nullable=False)
    vehicle_model = Column(String(64), nullable=False)
    vehicle_year = Column(Integer, nullable=False)
    vehicle_segment = Column(String(32), default="sedan")
    incident_description = Column(Text, default="")
    incident_date = Column(String(64), default="")
    status = Column(String(64), default="DRAFT", index=True)
    created_at = Column(String(64), default=utc_now_iso)
    updated_at = Column(String(64), default=utc_now_iso, onupdate=utc_now_iso)

    images = relationship("ImageDB", back_populates="claim", cascade="all, delete-orphan")
    assessment = relationship("AssessmentDB", back_populates="claim", uselist=False, cascade="all, delete-orphan")
    review = relationship("ReviewDB", back_populates="claim", uselist=False, cascade="all, delete-orphan")
    timeline = relationship("TimelineDB", back_populates="claim", cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "policy_number": self.policy_number,
            "vehicle_make": self.vehicle_make,
            "vehicle_model": self.vehicle_model,
            "vehicle_year": self.vehicle_year,
            "vehicle_segment": self.vehicle_segment,
            "incident_description": self.incident_description,
            "incident_date": self.incident_date,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "images": [img.to_dict() for img in self.images],
            "assessment": self.assessment.to_dict() if self.assessment else None,
            "review": self.review.to_dict() if self.review else None,
            "timeline": [evt.to_dict() for evt in self.timeline],
        }


class ImageDB(Base):
    __tablename__ = "claim_images"

    id = Column(String(64), primary_key=True, index=True)
    claim_id = Column(String(64), ForeignKey("claims.id", ondelete="CASCADE"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    local_path = Column(String(512), nullable=False)
    file_size_bytes = Column(Integer, default=0)
    mime_type = Column(String(64), default="image/jpeg")
    sha256 = Column(String(64), index=True, default="")
    width = Column(Integer, default=0)
    height = Column(Integer, default=0)
    created_at = Column(String(64), default=utc_now_iso)

    claim = relationship("ClaimDB", back_populates="images")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "claim_id": self.claim_id,
            "filename": self.filename,
            "local_path": self.local_path,
            "file_size_bytes": self.file_size_bytes,
            "mime_type": self.mime_type,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
            "created_at": self.created_at,
        }


class AssessmentDB(Base):
    __tablename__ = "assessments"

    id = Column(String(64), primary_key=True, index=True)
    claim_id = Column(String(64), ForeignKey("claims.id", ondelete="CASCADE"), unique=True, nullable=False)
    image_path = Column(String(512), default="")
    route = Column(String(64), default="TECHNICAL_REVIEW")
    reason_codes_json = Column(Text, default="[]")
    genai_gate_json = Column(Text, default="null")
    quality_json = Column(Text, default="null")
    fraud_json = Column(Text, default="null")
    severity_json = Column(Text, default="null")
    location_json = Column(Text, default="null")
    detections_json = Column(Text, default="[]")
    cost_json = Column(Text, default="null")
    model_versions_json = Column(Text, default="{}")
    inference_ms = Column(Integer, default=0)
    warnings_json = Column(Text, default="[]")
    created_at = Column(String(64), default=utc_now_iso)

    claim = relationship("ClaimDB", back_populates="assessment")

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "image_path": self.image_path,
            "route": self.route,
            "reason_codes": json.loads(self.reason_codes_json or "[]"),
            "genai_gate": json.loads(self.genai_gate_json) if hasattr(self, "genai_gate_json") and self.genai_gate_json and self.genai_gate_json != "null" else None,
            "quality": json.loads(self.quality_json) if self.quality_json and self.quality_json != "null" else None,
            "fraud": json.loads(self.fraud_json) if self.fraud_json and self.fraud_json != "null" else None,
            "severity": json.loads(self.severity_json) if self.severity_json and self.severity_json != "null" else None,
            "location": json.loads(self.location_json) if self.location_json and self.location_json != "null" else None,
            "detections": json.loads(self.detections_json or "[]"),
            "cost": json.loads(self.cost_json) if self.cost_json and self.cost_json != "null" else None,
            "model_versions": json.loads(self.model_versions_json or "{}"),
            "inference_ms": self.inference_ms,
            "warnings": json.loads(self.warnings_json or "[]"),
            "created_at": self.created_at,
        }


class ReviewDB(Base):
    __tablename__ = "reviews"

    id = Column(String(64), primary_key=True, index=True)
    claim_id = Column(String(64), ForeignKey("claims.id", ondelete="CASCADE"), unique=True, nullable=False)
    reviewer_id = Column(String(64), default="adjuster_1")
    decision = Column(String(64), default="PENDING")
    notes = Column(Text, default="")
    overrides_json = Column(Text, default="{}")
    timestamp = Column(String(64), default=utc_now_iso)

    claim = relationship("ClaimDB", back_populates="review")

    def to_dict(self) -> dict:
        return {
            "reviewer_id": self.reviewer_id,
            "decision": self.decision,
            "notes": self.notes,
            "overrides": json.loads(self.overrides_json or "{}"),
            "timestamp": self.timestamp,
        }


class TimelineDB(Base):
    __tablename__ = "timelines"

    id = Column(String(64), primary_key=True, index=True)
    claim_id = Column(String(64), ForeignKey("claims.id", ondelete="CASCADE"), nullable=False, index=True)
    from_status = Column(String(64), default="")
    to_status = Column(String(64), default="")
    reason = Column(Text, default="")
    actor = Column(String(64), default="SYSTEM")
    timestamp = Column(String(64), default=utc_now_iso)

    claim = relationship("ClaimDB", back_populates="timeline")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "claim_id": self.claim_id,
            "from_status": self.from_status,
            "to_status": self.to_status,
            "reason": self.reason,
            "actor": self.actor,
            "timestamp": self.timestamp,
        }
