"""Database schemas package (Step 12)."""
from src.database.schemas.camera import CameraBase, CameraCreate, CameraResponse
from src.database.schemas.journey import JourneyLegSchema, VehicleJourneySchema
from src.database.schemas.observation import (
    VehicleObservationBase,
    VehicleObservationCreate,
    VehicleObservationResponse,
)
from src.database.schemas.vehicle import (
    ObservedVehicleBase,
    ObservedVehicleCreate,
    ObservedVehicleResponse,
)

__all__ = [
    "CameraBase",
    "CameraCreate",
    "CameraResponse",
    "ObservedVehicleBase",
    "ObservedVehicleCreate",
    "ObservedVehicleResponse",
    "VehicleObservationBase",
    "VehicleObservationCreate",
    "VehicleObservationResponse",
    "VehicleJourneySchema",
    "JourneyLegSchema",
]
