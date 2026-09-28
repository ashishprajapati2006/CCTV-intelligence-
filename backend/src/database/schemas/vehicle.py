"""Observed Vehicle Validation and Transfer Schemas (Step 12)."""
from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ObservedVehicleBase(BaseModel):
    vehicle_id: str
    registration_number: str
    normalized_registration_number: str
    first_seen_at: Optional[str] = None
    last_seen_at: Optional[str] = None
    observation_count: int = 1
    camera_count: int = 1
    best_consensus_score: Optional[float] = None
    status: str = "CONFIRMED"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ObservedVehicleCreate(ObservedVehicleBase):
    pass


class ObservedVehicleResponse(ObservedVehicleBase):
    id: int
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    class Config:
        from_attributes = True
