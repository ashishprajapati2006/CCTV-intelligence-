"""Camera SQLAlchemy Model (Step 12).

Defines the cameras table storing:
- Core camera registry details
- Stream URLs (RTSP, HLS, WebRTC)
- Geographic coordinates and PostGIS Point geometry (SRID 4326)
- JSONB camera metadata and capabilities
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from geoalchemy2 import Geometry
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON

from src.database.connection import Base


class PointGeometry(TypeDecorator):
    """Geometry column type that resolves to PostGIS POINT(4326) on Postgres and Text on SQLite."""
    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect is not None and dialect.name == "postgresql":
            return dialect.type_descriptor(Geometry("POINT", srid=4326))
        return dialect.type_descriptor(Text())


class Camera(Base):
    """Authoritative camera registry entity with PostGIS spatial indexing."""
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(64), unique=True, nullable=False, index=True)
    camera_number = Column(Integer, nullable=True)
    name = Column(String(255), nullable=True)
    location = Column(Text, nullable=True)
    status = Column(String(32), nullable=True, default="active")

    codec = Column(String(32), nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    fps = Column(Float, nullable=True)
    bitrate = Column(Integer, nullable=True)

    rtsp_url = Column(Text, nullable=True)
    hls_url = Column(Text, nullable=True)
    webrtc_url = Column(Text, nullable=True)
    timezone = Column(String(64), nullable=True)

    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    # PostGIS Point(lon, lat, 4326)
    geom = Column(PointGeometry(), nullable=True)

    # JSONB metadata (stored in DB column named "metadata")
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
    last_seen_at = Column(DateTime(timezone=True), nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert camera to dictionary representation."""
        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "camera_number": self.camera_number,
            "name": self.name,
            "location": self.location,
            "status": self.status,
            "codec": self.codec,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "bitrate": self.bitrate,
            "rtsp_url": self.rtsp_url,
            "hls_url": self.hls_url,
            "webrtc_url": self.webrtc_url,
            "timezone": self.timezone,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "has_coordinates": self.latitude is not None and self.longitude is not None,
            "metadata": self.extra_metadata or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
        }
