"""Camera Validation and Transfer Schemas (Step 12)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CameraBase(BaseModel):
    camera_id: str
    camera_number: Optional[int] = None
    name: Optional[str] = None
    location: Optional[str] = None
    status: Optional[str] = "active"
    codec: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    bitrate: Optional[int] = None
    rtsp_url: Optional[str] = None
    hls_url: Optional[str] = None
    webrtc_url: Optional[str] = None
    timezone: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CameraCreate(CameraBase):
    pass


class CameraResponse(CameraBase):
    id: int
    has_coordinates: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_seen_at: Optional[str] = None

    class Config:
        from_attributes = True
