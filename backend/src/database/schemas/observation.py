"""Vehicle Observation Validation and Transfer Schemas (Step 12)."""
from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class VehicleObservationBase(BaseModel):
    observation_id: str
    vehicle_id: str
    camera_id: str
    track_id: str
    registration_number: str
    normalized_registration_number: str
    first_seen_pts_ms: float
    recognition_pts_ms: float
    last_seen_pts_ms: float
    source_time: Optional[str] = None
    source_time_status: str = "NOT_RESOLVED"
    source_time_method: Optional[str] = None
    source_time_confidence: Optional[float] = None
    loop_instance: int = 0
    media_session_id: Optional[str] = None
    recognition_status: str = "CONFIRMED"
    consensus_score: Optional[float] = None
    ocr_confidence: Optional[float] = None
    plate_detection_confidence: Optional[float] = None
    plate_quality_score: Optional[float] = None
    evidence_image: Optional[str] = None
    evidence_metadata: Dict[str, Any] = Field(default_factory=dict)


class VehicleObservationCreate(VehicleObservationBase):
    pass


class VehicleObservationResponse(VehicleObservationBase):
    id: int
    created_at: Optional[str] = None

    class Config:
        from_attributes = True
