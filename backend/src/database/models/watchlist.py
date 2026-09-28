"""Watchlist Entry SQLAlchemy Model (Step 12).

Defines watchlist_entries table storing hotlist targets with mandatory
synthetic flags for hackathon demonstration.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Index,
    Integer,
    String,
    Text,
)

from src.database.connection import Base


class WatchlistEntry(Base):
    """Hotlist target vehicle entry."""
    __tablename__ = "watchlist_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    watchlist_id = Column(String(64), unique=True, nullable=False, index=True)
    entity_type = Column(String(32), default="VEHICLE", nullable=False)

    registration_number = Column(String(32), nullable=False)
    normalized_registration_number = Column(
        String(32), nullable=False, index=True
    )

    category = Column(String(64), nullable=False, default="SUSPECT_VEHICLE")
    priority = Column(String(32), default="HIGH", nullable=False)
    status = Column(String(32), default="ACTIVE", nullable=False)

    source = Column(String(64), default="SYNTHETIC_DEMO", nullable=False)
    synthetic = Column(Boolean, default=True, nullable=False)

    description = Column(Text, nullable=True)

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
        """Convert watchlist target to dictionary."""
        return {
            "id": self.id,
            "watchlist_id": self.watchlist_id,
            "entity_type": self.entity_type,
            "registration_number": self.registration_number,
            "normalized_registration_number": self.normalized_registration_number,
            "category": self.category,
            "priority": self.priority,
            "status": self.status,
            "source": self.source,
            "synthetic": self.synthetic,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
