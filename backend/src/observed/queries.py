"""Query interface for Observed Vehicle Database (Step 7).

Provides high-level lookup functions for vehicles, cross-camera sightings,
chronological timelines, track provenance, and network-wide statistics.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from src.ai.anpr.normalizer import normalize_raw_text
from src.observed.models import ObservedVehicle, VehicleObservation, VehicleTrackRecord
from src.observed.repository import ObservedVehicleRepository, JSONObservedVehicleRepository


class ObservedVehicleQueries:
    """Service layer exposing query operations on ObservedVehicleRepository."""

    def __init__(self, repository: Optional[ObservedVehicleRepository] = None):
        self.repo = repository or JSONObservedVehicleRepository()

    def get_vehicle(self, registration_number: str) -> Optional[ObservedVehicle]:
        """Fetch aggregated vehicle profile by registration number."""
        norm = normalize_raw_text(registration_number)
        return self.repo.get_vehicle(norm)

    def get_vehicle_observations(self, registration_number: str) -> List[VehicleObservation]:
        """Fetch all individual observations supporting a vehicle."""
        norm = normalize_raw_text(registration_number)
        return self.repo.get_vehicle_observations(norm)

    def get_vehicle_cameras(self, registration_number: str) -> List[str]:
        """Fetch all distinct cameras where a vehicle was observed."""
        norm = normalize_raw_text(registration_number)
        return self.repo.get_vehicle_cameras(norm)

    def get_vehicle_timeline(self, registration_number: str) -> List[Dict[str, Any]]:
        """Fetch chronological observation history across all cameras."""
        norm = normalize_raw_text(registration_number)
        return self.repo.get_vehicle_timeline(norm)

    def get_track(self, track_id: str, camera_id: Optional[str] = None) -> Optional[VehicleTrackRecord]:
        """Fetch vehicle track record and supporting frame evidence."""
        return self.repo.get_track(track_id, camera_id=camera_id)

    def list_observed_vehicles(self, limit: int = 100, offset: int = 0) -> List[ObservedVehicle]:
        """List observed vehicles ordered by observation count."""
        return self.repo.list_vehicles(limit=limit, offset=offset)

    def count_observations(self) -> int:
        """Get total observation records count."""
        return self.repo.count_observations()

    def get_statistics(self) -> Dict[str, Any]:
        """Get network-wide observed vehicle statistics."""
        return self.repo.get_statistics()
