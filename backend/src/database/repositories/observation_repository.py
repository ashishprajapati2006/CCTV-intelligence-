"""Observation and ANPR Evidence Repository (Step 12).

Provides data access for:
- VehicleObservation records (track-level consensus sightings)
- ANPRObservation records (frame-level raw OCR evidence)
- VehicleTrack records
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.database.models.observation import ANPRObservation, VehicleObservation
from src.database.models.track import VehicleTrack


class ObservationRepository:
    """Repository managing vehicle observations, tracks, and ANPR evidence."""

    def __init__(self, session: Session):
        self.session = session

    # Vehicle Observations
    def get_observation(self, observation_id: str) -> Optional[VehicleObservation]:
        """Fetch observation by observation_id."""
        stmt = select(VehicleObservation).where(
            VehicleObservation.observation_id == observation_id
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def list_observations_by_vehicle(
        self, normalized_registration: str
    ) -> List[VehicleObservation]:
        """List all observations for a vehicle ordered by recognition PTS."""
        stmt = (
            select(VehicleObservation)
            .where(
                VehicleObservation.normalized_registration_number == normalized_registration
            )
            .order_by(VehicleObservation.camera_id, VehicleObservation.recognition_pts_ms)
        )
        return list(self.session.execute(stmt).scalars().all())

    def list_observations_by_camera(
        self, camera_id: str, limit: int = 100
    ) -> List[VehicleObservation]:
        """List observations recorded on a specific camera."""
        stmt = (
            select(VehicleObservation)
            .where(VehicleObservation.camera_id == camera_id)
            .order_by(VehicleObservation.recognition_pts_ms.desc())
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())

    def count_observations(self) -> int:
        """Total observation count."""
        stmt = select(func.count()).select_from(VehicleObservation)
        return self.session.execute(stmt).scalar() or 0

    def add_observation(self, data: Dict[str, Any]) -> VehicleObservation:
        """Add vehicle observation if not existing (idempotent)."""
        obs_id = data["observation_id"]
        existing = self.get_observation(obs_id)
        if existing:
            return existing

        obs = VehicleObservation(
            observation_id=obs_id,
            vehicle_id=data.get("vehicle_id", f"VEH-{data.get('normalized_registration_number')}"),
            camera_id=data["camera_id"],
            track_id=data["track_id"],
            registration_number=data["registration_number"],
            normalized_registration_number=data["normalized_registration_number"],
            first_seen_pts_ms=float(data["first_seen_pts_ms"]),
            recognition_pts_ms=float(data["recognition_pts_ms"]),
            last_seen_pts_ms=float(data["last_seen_pts_ms"]),
            source_time=data.get("source_time"),
            source_time_status=data.get("source_time_status", "NOT_RESOLVED"),
            source_time_method=data.get("source_time_method"),
            source_time_confidence=data.get("source_time_confidence"),
            loop_instance=data.get("loop_instance", 0),
            media_session_id=data.get("media_session_id"),
            recognition_status=data.get("recognition_status", data.get("status", "CONFIRMED")),
            consensus_score=data.get("consensus_score"),
            ocr_confidence=data.get("ocr_confidence"),
            plate_detection_confidence=data.get("plate_detection_confidence"),
            plate_quality_score=data.get("plate_quality_score"),
            evidence_image=data.get("evidence_image") or data.get("evidence_image_path"),
            evidence_metadata=data.get("evidence_metadata", {}),
        )
        self.session.add(obs)
        self.session.flush()
        return obs

    # ANPR Evidence
    def add_anpr_observation(self, data: Dict[str, Any]) -> ANPRObservation:
        """Add raw frame ANPR observation."""
        anpr = ANPRObservation(
            observation_id=data.get("observation_id"),
            camera_id=data["camera_id"],
            track_id=data["track_id"],
            frame_id=data.get("frame_id"),
            pts_ms=data.get("pts_ms"),
            raw_text=data.get("raw_text"),
            normalized_text=data.get("normalized_text"),
            ocr_confidence=data.get("ocr_confidence"),
            plate_detection_confidence=data.get("plate_detection_confidence"),
            plate_quality_score=data.get("plate_quality_score"),
            preprocessing_variant=data.get("preprocessing_variant"),
            image_crop_path=data.get("image_crop_path"),
            ocr_engine=data.get("ocr_engine", "EasyOCR"),
            ocr_engine_version=data.get("ocr_engine_version"),
        )
        self.session.add(anpr)
        self.session.flush()
        return anpr

    def count_anpr_observations(self) -> int:
        """Total frame-level ANPR observation count."""
        stmt = select(func.count()).select_from(ANPRObservation)
        return self.session.execute(stmt).scalar() or 0

    # Vehicle Tracks
    def save_track(self, data: Dict[str, Any]) -> VehicleTrack:
        """Save or update vehicle track record."""
        track_id = data["track_id"]
        camera_id = data["camera_id"]

        stmt = select(VehicleTrack).where(
            VehicleTrack.track_id == track_id, VehicleTrack.camera_id == camera_id
        )
        track = self.session.execute(stmt).scalar_one_or_none()

        if track is None:
            track = VehicleTrack(
                track_id=track_id,
                camera_id=camera_id,
                vehicle_class=data.get("vehicle_class", "vehicle"),
                first_seen_pts_ms=float(data["first_seen_pts_ms"]),
                last_seen_pts_ms=float(data["last_seen_pts_ms"]),
                first_source_time=data.get("first_source_time"),
                last_source_time=data.get("last_source_time"),
                frame_count=data.get("frame_count", 1),
                best_frame_paths=data.get("best_frame_paths", []),
            )
            self.session.add(track)
        else:
            track.last_seen_pts_ms = float(data.get("last_seen_pts_ms", track.last_seen_pts_ms))
            track.frame_count = data.get("frame_count", track.frame_count)
            if "best_frame_paths" in data:
                track.best_frame_paths = data["best_frame_paths"]

        self.session.flush()
        return track
