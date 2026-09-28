"""Observed Vehicle Repository (Step 12).

Provides data access for aggregated vehicle identities, multi-camera correlation,
and observation summary records.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.database.models.vehicle import ObservedVehicle


class VehicleRepository:
    """Repository handling observed vehicle persistence and aggregation queries."""

    def __init__(self, session: Session):
        self.session = session

    def get_vehicle(self, normalized_registration: str) -> Optional[ObservedVehicle]:
        """Fetch vehicle by normalized registration plate."""
        stmt = select(ObservedVehicle).where(
            ObservedVehicle.normalized_registration_number == normalized_registration
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def list_vehicles(self, limit: int = 100, offset: int = 0) -> List[ObservedVehicle]:
        """List observed vehicles ordered by observation count desc."""
        stmt = (
            select(ObservedVehicle)
            .order_by(ObservedVehicle.observation_count.desc(), ObservedVehicle.id.asc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.session.execute(stmt).scalars().all())

    def count_vehicles(self) -> int:
        """Total observed vehicle count."""
        stmt = select(func.count()).select_from(ObservedVehicle)
        return self.session.execute(stmt).scalar() or 0

    def find_multicamera_vehicles(self, min_cameras: int = 2) -> List[ObservedVehicle]:
        """Find vehicles observed on >= min_cameras distinct cameras."""
        stmt = (
            select(ObservedVehicle)
            .where(ObservedVehicle.camera_count >= min_cameras)
            .order_by(ObservedVehicle.camera_count.desc(), ObservedVehicle.observation_count.desc())
        )
        return list(self.session.execute(stmt).scalars().all())

    def upsert_vehicle(self, data: Dict[str, Any]) -> ObservedVehicle:
        """Insert or update observed vehicle record idempotently."""
        norm_reg = data["normalized_registration_number"]
        veh = self.get_vehicle(norm_reg)

        if veh is None:
            veh = ObservedVehicle(
                vehicle_id=data.get("vehicle_id", f"VEH-{norm_reg}"),
                registration_number=data.get("registration_number", norm_reg),
                normalized_registration_number=norm_reg,
                first_seen_at=data.get("first_seen_at"),
                last_seen_at=data.get("last_seen_at"),
                observation_count=data.get("observation_count", 1),
                camera_count=data.get("camera_count", 1),
                best_consensus_score=data.get("best_consensus_score"),
                status=data.get("status", "CONFIRMED"),
                extra_metadata=data.get("metadata", {}),
            )
            self.session.add(veh)
        else:
            veh.observation_count = data.get("observation_count", veh.observation_count)
            veh.camera_count = data.get("camera_count", veh.camera_count)
            if data.get("best_consensus_score") is not None:
                veh.best_consensus_score = max(
                    veh.best_consensus_score or 0.0,
                    float(data["best_consensus_score"]),
                )
            if data.get("first_seen_at"):
                veh.first_seen_at = data["first_seen_at"]
            if data.get("last_seen_at"):
                veh.last_seen_at = data["last_seen_at"]
            if "metadata" in data:
                veh.extra_metadata = data["metadata"]

        self.session.flush()
        return veh
