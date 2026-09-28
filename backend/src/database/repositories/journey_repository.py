"""Vehicle Journey Repository and GeoJSON Generator (Step 12).

Provides data access for:
- VehicleJourneyModel records
- JourneyLegModel transitions
- GeoJSON FeatureCollection generation adhering to Step 11 forensic rules
  (never fabricating road routes; only plotting verified camera locations).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from src.database.models.camera import Camera
from src.database.models.journey import (
    JourneyLegModel,
    JourneyObservationModel,
    VehicleJourneyModel,
)
from src.database.models.observation import VehicleObservation


def journey_to_geojson(
    journey_data: Dict[str, Any], camera_lookup: Optional[Dict[str, Camera]] = None
) -> Dict[str, Any]:
    """Convert vehicle journey observations to GeoJSON FeatureCollection.

    Forensic Rules (Part 12 & 21):
    - Only includes points where verified camera coordinates exist.
    - Explicitly marks lines as observation connection, NOT an inferred road route.
    """
    features = []
    coord_points = []

    observations = journey_data.get("observations") or journey_data.get("segments") or []
    reg = journey_data.get("registration_number", "UNKNOWN")

    for i, obs in enumerate(observations, 1):
        cam_id = obs.get("camera_id")
        lat = obs.get("camera_latitude")
        lon = obs.get("camera_longitude")

        # Check camera lookup if coordinates missing on segment
        if (lat is None or lon is None) and camera_lookup and cam_id in camera_lookup:
            cam = camera_lookup[cam_id]
            lat = cam.latitude
            lon = cam.longitude

        if lat is not None and lon is not None:
            point_coords = [float(lon), float(lat)]
            coord_points.append(point_coords)

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": point_coords,
                },
                "properties": {
                    "layer": "observation_point",
                    "camera_id": cam_id,
                    "registration_number": reg,
                    "sequence": i,
                    "source_time": obs.get("source_time"),
                    "source_time_status": obs.get("source_time_status", "NOT_RESOLVED"),
                    "recognition_pts_ms": obs.get("recognition_pts_ms"),
                    "consensus_score": obs.get("consensus_score"),
                    "evidence_image": obs.get("evidence_image") or obs.get("evidence_image_path"),
                },
            }
            features.append(feature)

    # If 2+ verified points exist, add straight-line connecting trajectory with disclaimer
    if len(coord_points) >= 2:
        line_feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coord_points,
            },
            "properties": {
                "layer": "trajectory",
                "registration_number": reg,
                "point_count": len(coord_points),
                "disclaimer": "Approximate straight-line camera observation connection; NOT an actual road route.",
            },
        }
        features.append(line_feature)

    return {
        "type": "FeatureCollection",
        "features": features,
    }


class JourneyRepository:
    """Repository handling vehicle journey persistence and spatial GeoJSON retrieval."""

    def __init__(self, session: Session):
        self.session = session

    def get_journey(self, registration_number: str) -> Optional[Dict[str, Any]]:
        """Retrieve journey header, observations, and legs by registration number."""
        stmt = select(VehicleJourneyModel).where(
            VehicleJourneyModel.registration_number == registration_number
        )
        journey = self.session.execute(stmt).scalar_one_or_none()
        if not journey:
            return None

        # Fetch legs
        legs_stmt = (
            select(JourneyLegModel)
            .where(JourneyLegModel.journey_id == journey.journey_id)
            .order_by(JourneyLegModel.id.asc())
        )
        legs = list(self.session.execute(legs_stmt).scalars().all())

        # Fetch observation mappings
        obs_map_stmt = (
            select(JourneyObservationModel)
            .where(JourneyObservationModel.journey_id == journey.journey_id)
            .order_by(JourneyObservationModel.sequence_number.asc())
        )
        obs_maps = list(self.session.execute(obs_map_stmt).scalars().all())

        # Fetch actual observation details
        obs_ids = [m.observation_id for m in obs_maps]
        obs_list = []
        if obs_ids:
            obs_stmt = select(VehicleObservation).where(
                VehicleObservation.observation_id.in_(obs_ids)
            )
            obs_dict = {o.observation_id: o for o in self.session.execute(obs_stmt).scalars().all()}
            # Fetch camera coordinates
            cam_ids = {o.camera_id for o in obs_dict.values()}
            cam_stmt = select(Camera).where(Camera.camera_id.in_(cam_ids))
            cams = {c.camera_id: c for c in self.session.execute(cam_stmt).scalars().all()}

            for m in obs_maps:
                o = obs_dict.get(m.observation_id)
                if o:
                    d = o.to_dict()
                    cam = cams.get(o.camera_id)
                    d["camera_latitude"] = cam.latitude if cam else None
                    d["camera_longitude"] = cam.longitude if cam else None
                    d["sequence_number"] = m.sequence_number
                    obs_list.append(d)

        return {
            **journey.to_dict(),
            "observations": obs_list,
            "legs": [leg.to_dict() for leg in legs],
        }

    def list_journeys(self, limit: int = 50) -> List[VehicleJourneyModel]:
        """List stored vehicle journeys."""
        stmt = (
            select(VehicleJourneyModel)
            .order_by(VehicleJourneyModel.updated_at.desc())
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())

    def count(self) -> int:
        """Total journey records count."""
        stmt = select(func.count()).select_from(VehicleJourneyModel)
        return self.session.execute(stmt).scalar() or 0

    def save_journey_data(self, data: Dict[str, Any]) -> VehicleJourneyModel:
        """Persist or update journey, legs, and observation sequence idempotently."""
        jid = data["journey_id"]
        reg = data["registration_number"]

        stmt = select(VehicleJourneyModel).where(VehicleJourneyModel.journey_id == jid)
        journey = self.session.execute(stmt).scalar_one_or_none()

        if journey is None:
            journey = VehicleJourneyModel(
                journey_id=jid,
                vehicle_id=data.get("vehicle_id", f"VEH-{reg}"),
                registration_number=reg,
                status=data.get("status", "OBSERVATION_SEQUENCE_ONLY"),
                time_basis=data.get("time_basis", "CAMERA_LOCAL_MEDIA_PTS"),
                source_time_resolution_status=data.get("source_time_resolution_status", "NOT_RESOLVED"),
                confidence=float(data.get("confidence", 0.0)),
            )
            self.session.add(journey)
        else:
            journey.status = data.get("status", journey.status)
            journey.time_basis = data.get("time_basis", journey.time_basis)
            journey.source_time_resolution_status = data.get("source_time_resolution_status", journey.source_time_resolution_status)
            journey.confidence = float(data.get("confidence", journey.confidence))

        # Clear existing legs and observation sequence for clean update
        self.session.execute(delete(JourneyLegModel).where(JourneyLegModel.journey_id == jid))
        self.session.execute(delete(JourneyObservationModel).where(JourneyObservationModel.journey_id == jid))

        # Insert observation mapping
        observations = data.get("observations") or data.get("segments") or []
        for seq, obs in enumerate(observations, 1):
            obs_id = obs.get("observation_id") or obs.get("segment_id", f"OBS-{seq}")
            self.session.add(
                JourneyObservationModel(
                    journey_id=jid,
                    observation_id=obs_id,
                    sequence_number=seq,
                )
            )

        # Insert legs
        legs = data.get("legs", [])
        for leg in legs:
            self.session.add(
                JourneyLegModel(
                    journey_id=jid,
                    from_observation_id=leg.get("from_observation_id", ""),
                    to_observation_id=leg.get("to_observation_id", ""),
                    from_camera_id=leg.get("from_camera_id", ""),
                    to_camera_id=leg.get("to_camera_id", ""),
                    from_source_time=leg.get("from_source_time"),
                    to_source_time=leg.get("to_source_time"),
                    time_delta_seconds=leg.get("time_delta_seconds"),
                    straight_line_distance_m=leg.get("straight_line_distance_m"),
                    distance_status=leg.get("distance_status", "UNAVAILABLE"),
                    plausibility=leg.get("plausibility", "UNKNOWN"),
                    implied_straight_line_speed=leg.get("implied_straight_line_speed"),
                )
            )

        self.session.flush()
        return journey
