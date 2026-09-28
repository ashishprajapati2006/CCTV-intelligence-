"""Database models package (Step 12).

Exports all declarative SQLAlchemy models for easy access and Alembic discovery.
"""
from src.database.models.alert import AlertModel
from src.database.models.camera import Camera
from src.database.models.journey import (
    JourneyLegModel,
    JourneyObservationModel,
    VehicleJourneyModel,
)
from src.database.models.match import WatchlistMatch
from src.database.models.observation import ANPRObservation, VehicleObservation
from src.database.models.track import VehicleTrack
from src.database.models.vehicle import ObservedVehicle
from src.database.models.watchlist import WatchlistEntry

__all__ = [
    "Camera",
    "ObservedVehicle",
    "VehicleObservation",
    "ANPRObservation",
    "VehicleTrack",
    "WatchlistEntry",
    "WatchlistMatch",
    "AlertModel",
    "VehicleJourneyModel",
    "JourneyObservationModel",
    "JourneyLegModel",
]
