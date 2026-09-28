"""Repositories package for CCTV Intelligence Platform (Step 12).

Exports decoupled repository classes for cameras, vehicles, observations,
watchlists, matches, alerts, and journeys.
"""
from src.database.repositories.alert_repository import AlertRepository
from src.database.repositories.camera_repository import CameraRepository
from src.database.repositories.journey_repository import (
    JourneyRepository,
    journey_to_geojson,
)
from src.database.repositories.match_repository import MatchRepository
from src.database.repositories.observation_repository import ObservationRepository
from src.database.repositories.vehicle_repository import VehicleRepository
from src.database.repositories.watchlist_repository import WatchlistRepository

__all__ = [
    "CameraRepository",
    "VehicleRepository",
    "ObservationRepository",
    "WatchlistRepository",
    "MatchRepository",
    "AlertRepository",
    "JourneyRepository",
    "journey_to_geojson",
]
