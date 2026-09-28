"""Observation Loader and Camera Catalogue Join for Step 11.

Loads vehicle observations from Step 7 and joins with camera catalogue metadata
(coordinates, location name) without inventing coordinates.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.logging import get_logger
from src.journey.models import CrossCameraObservation
from src.watchlist.normalizer import normalize_watchlist_registration

logger = get_logger("journey_loader")


class ObservationLoader:
    """Loads observations from Step 7 and enriches with camera catalogue metadata."""

    def __init__(
        self,
        observations_file: str = "data/observed/vehicles/observations.jsonl",
        vehicles_file: str = "data/observed/vehicles/vehicles.json",
        cameras_file: str = "data/catalogue/normalized/cameras.json",
    ):
        self.observations_path = Path(observations_file)
        self.vehicles_path = Path(vehicles_file)
        self.cameras_path = Path(cameras_file)
        self._cameras_cache: Dict[str, Dict[str, Any]] = {}
        self._vehicle_ids_cache: Dict[str, str] = {}  # norm_reg -> vehicle_id
        self._load_camera_catalogue()
        self._load_vehicle_index()

    def _load_camera_catalogue(self) -> None:
        if self.cameras_path.exists():
            try:
                with open(self.cameras_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for cam in data:
                            cid = cam.get("camera_id")
                            if cid:
                                self._cameras_cache[cid] = cam
            except Exception as e:
                logger.error("Failed to load cameras catalogue: %s", e)

    def _load_vehicle_index(self) -> None:
        if self.vehicles_path.exists():
            try:
                with open(self.vehicles_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for v in data:
                            reg = v.get("normalized_registration_number")
                            vid = v.get("vehicle_id")
                            if reg and vid:
                                self._vehicle_ids_cache[reg] = vid
            except Exception as e:
                logger.error("Failed to load vehicles index: %s", e)

    def load_observations_for_vehicle(
        self,
        registration_number: str,
        include_probable: bool = False,
    ) -> List[CrossCameraObservation]:
        """Retrieve all qualifying observations for a registration number."""
        target_norm = normalize_watchlist_registration(registration_number)
        observations: List[CrossCameraObservation] = []

        if not self.observations_path.exists():
            logger.warning("Observations file does not exist: %s", self.observations_path)
            return []

        allowed_statuses = {"CONFIRMED"}
        if include_probable:
            allowed_statuses.add("PROBABLE")

        with open(self.observations_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    rec_norm = record.get("normalized_registration_number") or normalize_watchlist_registration(record.get("registration_number", ""))
                    if rec_norm != target_norm:
                        continue

                    status = record.get("status", "CONFIRMED")
                    if status not in allowed_statuses:
                        continue

                    cam_id = record.get("camera_id", "unknown")
                    cam_meta = self._cameras_cache.get(cam_id, {})
                    vehicle_id = self._vehicle_ids_cache.get(target_norm, f"VEH-{target_norm}")

                    obs = CrossCameraObservation(
                        observation_id=record.get("observation_id", ""),
                        vehicle_id=vehicle_id,
                        registration_number=record.get("registration_number", target_norm),
                        normalized_registration_number=target_norm,
                        camera_id=cam_id,
                        track_id=record.get("track_id", "unknown"),
                        first_seen_pts_ms=record.get("first_seen_pts_ms") or record.get("observed_at_pts_ms"),
                        recognition_pts_ms=record.get("recognition_pts_ms"),
                        last_seen_pts_ms=record.get("last_seen_pts_ms"),
                        source_time=record.get("source_time"),
                        source_time_status=record.get("source_time_status", "NOT_RESOLVED"),
                        loop_instance=record.get("loop_instance"),
                        recognition_status=status,
                        consensus_score=float(record.get("consensus_score", 0.0)),
                        ocr_confidence=float(record.get("ocr_confidence", 0.0)),
                        camera_name=cam_meta.get("name"),
                        camera_latitude=cam_meta.get("latitude"),
                        camera_longitude=cam_meta.get("longitude"),
                        evidence_image=record.get("evidence_image"),
                        evidence_filename_pts_status=record.get("evidence_filename_pts_status", "UNKNOWN"),
                        source=record.get("source", "sentinel"),
                        ingested_at_utc=record.get("ingested_at_utc"),
                    )
                    observations.append(obs)

                except Exception as e:
                    logger.error("Failed to parse observation line: %s", e)

        return observations

    def list_all_observed_registrations(self) -> List[str]:
        """Return list of all unique normalized registrations present in observations.jsonl."""
        regs = set()
        if self.observations_path.exists():
            with open(self.observations_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        norm = record.get("normalized_registration_number")
                        if norm:
                            regs.add(norm)
                    except Exception:
                        pass
        return sorted(list(regs))
