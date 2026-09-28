"""Observed Vehicle SQLAlchemy Model (Step 12).

Defines observed_vehicles table storing unique recognized vehicle entities
aggregated across tracks and cameras.
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
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON

from src.database.connection import Base


class ObservedVehicle(Base):
    """Authoritative observed vehicle entity."""
    __tablename__ = "observed_vehicles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(String(64), nullable=False, index=True)
    registration_number = Column(String(32), nullable=False)
    normalized_registration_number = Column(
        String(32), unique=True, nullable=False, index=True
    )

    # Valid source-time calendar anchors if resolved; nullable if only PTS is available
    first_seen_at = Column(DateTime(timezone=True), nullable=True)
    last_seen_at = Column(DateTime(timezone=True), nullable=True)

    observation_count = Column(Integer, default=1, nullable=False)
    camera_count = Column(Integer, default=1, nullable=False)
    best_consensus_score = Column(Float, nullable=True)
    status = Column(String(32), default="CONFIRMED", nullable=False)

    extra_metadata = Column("metadata", JSON().with_variant(JSONB, "postgresql"), nullable=True, default=dict)

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
        """Convert vehicle to dictionary."""
        return {
            "id": self.id,
            "vehicle_id": self.vehicle_id,
            "registration_number": self.registration_number,
            "normalized_registration_number": self.normalized_registration_number,
            "first_seen_at": self.first_seen_at.isoformat() if self.first_seen_at else None,
            "last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
            "observation_count": self.observation_count,
            "camera_count": self.camera_count,
            "best_consensus_score": self.best_consensus_score,
            "status": self.status,
            "metadata": self.extra_metadata or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
