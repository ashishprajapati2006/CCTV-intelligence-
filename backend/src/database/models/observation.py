"""Vehicle and ANPR Observation SQLAlchemy Models (Step 12).

Defines:
1. vehicle_observations: Track-level consensus ANPR sightings across cameras.
2. anpr_observations: Per-frame OCR detection evidence preserving the raw recognition chain.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON

from src.database.connection import Base


class VehicleObservation(Base):
    """Observation entity representing a recognized vehicle track on a specific camera."""
    __tablename__ = "vehicle_observations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    observation_id = Column(String(64), unique=True, nullable=False, index=True)
    vehicle_id = Column(String(64), nullable=False, index=True)
    camera_id = Column(String(64), nullable=False, index=True)
    track_id = Column(String(64), nullable=False, index=True)

    registration_number = Column(String(32), nullable=False)
    normalized_registration_number = Column(
        String(32), nullable=False, index=True
    )

    # Media timing (PTS strictly camera-local)
    first_seen_pts_ms = Column(Float, nullable=False)
    recognition_pts_ms = Column(Float, nullable=False, index=True)
    last_seen_pts_ms = Column(Float, nullable=False)

    # Source calendar time (NTP / RTSP wall-clock)
    source_time = Column(DateTime(timezone=True), nullable=True, index=True)
    source_time_status = Column(String(32), default="NOT_RESOLVED", nullable=False)
    source_time_method = Column(String(64), nullable=True)
    source_time_confidence = Column(Float, nullable=True)

    loop_instance = Column(Integer, default=0, nullable=False)
    media_session_id = Column(String(64), nullable=True)

    recognition_status = Column(String(32), default="CONFIRMED", nullable=False)
    consensus_score = Column(Float, nullable=True)
    ocr_confidence = Column(Float, nullable=True)
    plate_detection_confidence = Column(Float, nullable=True)
    plate_quality_score = Column(Float, nullable=True)

    evidence_image = Column(Text, nullable=True)
    evidence_metadata = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True, default=dict)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert observation to dictionary."""
        return {
            "id": self.id,
            "observation_id": self.observation_id,
            "vehicle_id": self.vehicle_id,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "registration_number": self.registration_number,
            "normalized_registration_number": self.normalized_registration_number,
            "first_seen_pts_ms": self.first_seen_pts_ms,
            "recognition_pts_ms": self.recognition_pts_ms,
            "last_seen_pts_ms": self.last_seen_pts_ms,
            "source_time": self.source_time.isoformat() if self.source_time else None,
            "source_time_status": self.source_time_status,
            "source_time_method": self.source_time_method,
            "source_time_confidence": self.source_time_confidence,
            "loop_instance": self.loop_instance,
            "media_session_id": self.media_session_id,
            "recognition_status": self.recognition_status,
            "consensus_score": self.consensus_score,
            "ocr_confidence": self.ocr_confidence,
            "plate_detection_confidence": self.plate_detection_confidence,
            "plate_quality_score": self.plate_quality_score,
            "evidence_image": self.evidence_image,
            "evidence_metadata": self.evidence_metadata or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ANPRObservation(Base):
    """Raw frame-level ANPR recognition evidence preserving OCR provenance."""
    __tablename__ = "anpr_observations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    observation_id = Column(String(64), nullable=True, index=True)
    camera_id = Column(String(64), nullable=False, index=True)
    track_id = Column(String(64), nullable=False, index=True)
    frame_id = Column(Integer, nullable=True)
    pts_ms = Column(Float, nullable=True)

    raw_text = Column(String(64), nullable=True)
    normalized_text = Column(String(32), nullable=True, index=True)
    ocr_confidence = Column(Float, nullable=True)

    plate_detection_confidence = Column(Float, nullable=True)
    plate_quality_score = Column(Float, nullable=True)
    preprocessing_variant = Column(String(64), nullable=True)

    image_crop_path = Column(Text, nullable=True)
    ocr_engine = Column(String(64), default="EasyOCR", nullable=False)
    ocr_engine_version = Column(String(32), nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert ANPR observation to dictionary."""
        return {
            "id": self.id,
            "observation_id": self.observation_id,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "frame_id": self.frame_id,
            "pts_ms": self.pts_ms,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "ocr_confidence": self.ocr_confidence,
            "plate_detection_confidence": self.plate_detection_confidence,
            "plate_quality_score": self.plate_quality_score,
            "preprocessing_variant": self.preprocessing_variant,
            "image_crop_path": self.image_crop_path,
            "ocr_engine": self.ocr_engine,
            "ocr_engine_version": self.ocr_engine_version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
