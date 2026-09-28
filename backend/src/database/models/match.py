"""Watchlist Match SQLAlchemy Model (Step 12).

Defines watchlist_matches table storing real-time alert triggers
correlating observed vehicles against active hotlist entries.
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

from src.database.connection import Base


class WatchlistMatch(Base):
    """Verified match between an observed vehicle and a watchlist target."""
    __tablename__ = "watchlist_matches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(64), unique=True, nullable=False, index=True)
    observation_id = Column(String(64), nullable=False, index=True)
    watchlist_id = Column(String(64), nullable=False, index=True)

    decision = Column(String(32), default="MATCH", nullable=False)
    decision_reason = Column(Text, nullable=True)

    registration_number = Column(String(32), nullable=False)
    normalized_registration_number = Column(
        String(32), nullable=False, index=True
    )

    match_score = Column(Float, nullable=False)
    recognition_confidence = Column(Float, nullable=True)
    consensus_score = Column(Float, nullable=True)

    camera_id = Column(String(64), nullable=False, index=True)
    track_id = Column(String(64), nullable=False, index=True)

    recognition_pts_ms = Column(Float, nullable=False)
    source_time = Column(DateTime(timezone=True), nullable=True, index=True)
    source_time_status = Column(String(32), default="NOT_RESOLVED", nullable=False)

    evidence_image = Column(Text, nullable=True)

    watchlist_category = Column(String(64), nullable=True)
    watchlist_priority = Column(String(32), nullable=True)
    watchlist_status = Column(String(32), nullable=True)
    watchlist_source = Column(String(64), default="SYNTHETIC_DEMO", nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert match record to dictionary."""
        return {
            "id": self.id,
            "match_id": self.match_id,
            "observation_id": self.observation_id,
            "watchlist_id": self.watchlist_id,
            "decision": self.decision,
            "decision_reason": self.decision_reason,
            "registration_number": self.registration_number,
            "normalized_registration_number": self.normalized_registration_number,
            "match_score": self.match_score,
            "recognition_confidence": self.recognition_confidence,
            "consensus_score": self.consensus_score,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "recognition_pts_ms": self.recognition_pts_ms,
            "source_time": self.source_time.isoformat() if self.source_time else None,
            "source_time_status": self.source_time_status,
            "evidence_image": self.evidence_image,
            "watchlist_category": self.watchlist_category,
            "watchlist_priority": self.watchlist_priority,
            "watchlist_status": self.watchlist_status,
            "watchlist_source": self.watchlist_source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
