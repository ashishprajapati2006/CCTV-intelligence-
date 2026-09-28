"""Vehicle Journey SQLAlchemy Models (Step 12).

Defines:
1. vehicle_journeys: Journey aggregate headers with confidence and time basis.
2. journey_observations: Ordered observation sequences per vehicle journey.
3. journey_legs: Transitions between consecutive observations with Haversine distance,
   implied speed, and plausibility status.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)

from src.database.connection import Base


class VehicleJourneyModel(Base):
    """Reconstructed vehicle journey entity."""
    __tablename__ = "vehicle_journeys"

    id = Column(Integer, primary_key=True, autoincrement=True)
    journey_id = Column(String(64), unique=True, nullable=False, index=True)
    vehicle_id = Column(String(64), nullable=False)
    registration_number = Column(String(32), nullable=False, index=True)

    status = Column(String(32), default="OBSERVATION_SEQUENCE_ONLY", nullable=False)
    time_basis = Column(String(32), default="CAMERA_LOCAL_MEDIA_PTS", nullable=False)
    source_time_resolution_status = Column(String(32), default="NOT_RESOLVED", nullable=False)

    confidence = Column(Float, nullable=False, default=0.0)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert journey to dictionary."""
        return {
            "id": self.id,
            "journey_id": self.journey_id,
            "vehicle_id": self.vehicle_id,
            "registration_number": self.registration_number,
            "status": self.status,
            "time_basis": self.time_basis,
            "source_time_resolution_status": self.source_time_resolution_status,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class JourneyObservationModel(Base):
    """Observation mapping within an ordered vehicle journey sequence."""
    __tablename__ = "journey_observations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    journey_id = Column(String(64), nullable=False, index=True)
    observation_id = Column(String(64), nullable=False, index=True)
    sequence_number = Column(Integer, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "journey_id": self.journey_id,
            "observation_id": self.observation_id,
            "sequence_number": self.sequence_number,
        }


class JourneyLegModel(Base):
    """Inter-camera transition leg with physical plausibility and distance."""
    __tablename__ = "journey_legs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    journey_id = Column(String(64), nullable=False, index=True)

    from_observation_id = Column(String(64), nullable=False)
    to_observation_id = Column(String(64), nullable=False)

    from_camera_id = Column(String(64), nullable=False)
    to_camera_id = Column(String(64), nullable=False)

    from_source_time = Column(DateTime(timezone=True), nullable=True)
    to_source_time = Column(DateTime(timezone=True), nullable=True)

    time_delta_seconds = Column(Float, nullable=True)
    straight_line_distance_m = Column(Float, nullable=True)
    distance_status = Column(String(32), default="UNAVAILABLE", nullable=False)

    plausibility = Column(String(32), default="UNKNOWN", nullable=False)
    implied_straight_line_speed = Column(Float, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "journey_id": self.journey_id,
            "from_observation_id": self.from_observation_id,
            "to_observation_id": self.to_observation_id,
            "from_camera_id": self.from_camera_id,
            "to_camera_id": self.to_camera_id,
            "from_source_time": self.from_source_time.isoformat() if self.from_source_time else None,
            "to_source_time": self.to_source_time.isoformat() if self.to_source_time else None,
            "time_delta_seconds": self.time_delta_seconds,
            "straight_line_distance_m": self.straight_line_distance_m,
            "distance_status": self.distance_status,
            "plausibility": self.plausibility,
            "implied_straight_line_speed": self.implied_straight_line_speed,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
