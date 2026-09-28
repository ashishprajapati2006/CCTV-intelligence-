"""Vehicle Track SQLAlchemy Model (Step 12).

Defines vehicle_tracks table preserving multi-object tracking history
and frame associations per camera stream.
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


class VehicleTrack(Base):
    """Multi-object tracking record on a camera stream."""
    __tablename__ = "vehicle_tracks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(String(64), nullable=False, index=True)
    camera_id = Column(String(64), nullable=False, index=True)
    vehicle_class = Column(String(32), nullable=True, default="vehicle")

    first_seen_pts_ms = Column(Float, nullable=False)
    last_seen_pts_ms = Column(Float, nullable=False)

    first_source_time = Column(DateTime(timezone=True), nullable=True)
    last_source_time = Column(DateTime(timezone=True), nullable=True)

    frame_count = Column(Integer, default=1, nullable=False)
    best_frame_paths = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True, default=list)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert track record to dictionary."""
        return {
            "id": self.id,
            "track_id": self.track_id,
            "camera_id": self.camera_id,
            "vehicle_class": self.vehicle_class,
            "first_seen_pts_ms": self.first_seen_pts_ms,
            "last_seen_pts_ms": self.last_seen_pts_ms,
            "first_source_time": self.first_source_time.isoformat() if self.first_source_time else None,
            "last_source_time": self.last_source_time.isoformat() if self.last_source_time else None,
            "frame_count": self.frame_count,
            "best_frame_paths": self.best_frame_paths or [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
