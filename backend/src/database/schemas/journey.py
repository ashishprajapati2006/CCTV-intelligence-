"""Vehicle Journey Validation and Transfer Schemas (Step 12)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class JourneyLegSchema(BaseModel):
    from_observation_id: str
    to_observation_id: str
    from_camera_id: str
    to_camera_id: str
    from_source_time: Optional[str] = None
    to_source_time: Optional[str] = None
    time_delta_seconds: Optional[float] = None
    straight_line_distance_m: Optional[float] = None
    distance_status: str = "UNAVAILABLE"
    plausibility: str = "UNKNOWN"
    implied_straight_line_speed: Optional[float] = None


class VehicleJourneySchema(BaseModel):
    journey_id: str
    vehicle_id: str
    registration_number: str
    status: str = "OBSERVATION_SEQUENCE_ONLY"
    time_basis: str = "CAMERA_LOCAL_MEDIA_PTS"
    source_time_resolution_status: str = "NOT_RESOLVED"
    confidence: float = 0.0
    legs: List[JourneyLegSchema] = Field(default_factory=list)
    created_at: Optional[str] = None
