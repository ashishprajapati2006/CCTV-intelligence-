"""Incident Alert SQLAlchemy Model (Step 12).

Defines alerts table tracking operator acknowledgment, escalation,
and resolution workflows.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Index,
    Integer,
    String,
    Text,
)

from src.database.connection import Base


class AlertModel(Base):
    """Incident alert entity representing actionable operator notifications."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String(64), unique=True, nullable=False, index=True)
    match_id = Column(String(64), nullable=True, index=True)

    alert_type = Column(String(32), default="WATCHLIST_HIT", nullable=False)
    priority = Column(String(32), default="HIGH", nullable=False)
    status = Column(String(32), default="NEW", nullable=False)

    registration_number = Column(String(32), nullable=False, index=True)
    camera_id = Column(String(64), nullable=False, index=True)
    track_id = Column(String(64), nullable=False)

    message = Column(Text, nullable=True)

    source_time = Column(DateTime(timezone=True), nullable=True)
    source_time_status = Column(String(32), default="NOT_RESOLVED", nullable=False)

    evidence_image = Column(Text, nullable=True)

    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)

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
        """Convert alert record to dictionary."""
        return {
            "id": self.id,
            "alert_id": self.alert_id,
            "match_id": self.match_id,
            "alert_type": self.alert_type,
            "priority": self.priority,
            "status": self.status,
            "registration_number": self.registration_number,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "message": self.message,
            "source_time": self.source_time.isoformat() if self.source_time else None,
            "source_time_status": self.source_time_status,
            "evidence_image": self.evidence_image,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
