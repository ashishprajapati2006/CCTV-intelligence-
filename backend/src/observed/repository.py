"""Repository layer for Observed Vehicle Database (Step 7).

Defines:
  1. ObservedVehicleRepository: Abstract interface ready for future PostgreSQL swap.
  2. JSONObservedVehicleRepository: Concrete JSON/JSONL implementation for current PoC.

Storage Layout:
  data/observed/vehicles/
    - observations.jsonl
    - vehicles.json
    - tracks.json
    - uncertain_observations.jsonl
"""
from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.common.logging import get_logger
from src.observed.models import (
    ObservedVehicle,
    UncertainObservation,
    VehicleObservation,
    VehicleTrackRecord,
)

logger = get_logger("observed_repo")


class ObservedVehicleRepository(abc.ABC):
    """Abstract base repository for observed vehicles, tracks, and observations."""

    @abc.abstractmethod
    def add_observation(self, observation: VehicleObservation) -> bool:
        """Add an observation if not duplicate. Returns True if added."""
        pass

    @abc.abstractmethod
    def add_uncertain_observation(self, observation: UncertainObservation) -> bool:
        """Add an uncertain/unreadable observation record."""
        pass

    @abc.abstractmethod
    def save_vehicle(self, vehicle: ObservedVehicle) -> None:
        """Save or update aggregated vehicle record."""
        pass

    @abc.abstractmethod
    def save_track(self, track: VehicleTrackRecord) -> None:
        """Save or update vehicle track record."""
        pass

    @abc.abstractmethod
    def get_vehicle(self, normalized_registration: str) -> Optional[ObservedVehicle]:
        """Fetch aggregated vehicle by normalized registration number."""
        pass

    @abc.abstractmethod
    def get_vehicle_observations(self, normalized_registration: str) -> List[VehicleObservation]:
        """Fetch all individual observations for a vehicle."""
        pass

    @abc.abstractmethod
    def get_vehicle_cameras(self, normalized_registration: str) -> List[str]:
        """Fetch distinct camera IDs where a vehicle was observed."""
        pass

    @abc.abstractmethod
    def get_vehicle_timeline(self, normalized_registration: str) -> List[Dict[str, Any]]:
        """Fetch chronological sighting timeline for a vehicle."""
        pass

    @abc.abstractmethod
    def get_track(self, track_id: str, camera_id: Optional[str] = None) -> Optional[VehicleTrackRecord]:
        """Fetch track record by track_id (and optional camera_id)."""
        pass

    @abc.abstractmethod
    def list_vehicles(self, limit: int = 100, offset: int = 0) -> List[ObservedVehicle]:
        """List observed vehicles."""
        pass

    @abc.abstractmethod
    def count_observations(self) -> int:
        """Total observation records count."""
        pass

    @abc.abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Retrieve network-wide observed vehicle statistics."""
        pass


def _safe_replace(src: Path, dst: Path, max_retries: int = 5) -> None:
    import shutil
    import time
    for attempt in range(max_retries):
        try:
            src.replace(dst)
            return
        except (PermissionError, OSError):
            if attempt == max_retries - 1:
                try:
                    shutil.copyfile(src, dst)
                    if src.exists():
                        src.unlink()
                except Exception:
                    pass
                return
            time.sleep(0.05 * (attempt + 1))


class JSONObservedVehicleRepository(ObservedVehicleRepository):
    """Concrete file-based storage using JSONL and JSON files.

    Maintains in-memory indices for fast lookups while persisting to disk atomically.
    Ensures idempotent operations with deterministic duplicate prevention.
    """

    def __init__(self, storage_dir: str = "data/observed/vehicles"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.observations_file = self.storage_dir / "observations.jsonl"
        self.vehicles_file = self.storage_dir / "vehicles.json"
        self.tracks_file = self.storage_dir / "tracks.json"
        self.uncertain_file = self.storage_dir / "uncertain_observations.jsonl"

        # In-memory indices
        self._vehicles: Dict[str, ObservedVehicle] = {}  # normalized_reg -> ObservedVehicle
        self._tracks: Dict[str, VehicleTrackRecord] = {}    # f"{cam_id}_{track_id}" -> VehicleTrackRecord
        self._observations: List[VehicleObservation] = []
        self._uncertain: List[UncertainObservation] = []
        self._observation_keys: Set[str] = set()  # composite keys for deduplication
        self._uncertain_keys: Set[str] = set()

        self._load_from_disk()

    @staticmethod
    def _make_obs_key(
        camera_id: str,
        track_id: str,
        norm_reg: str,
        pts_ms: Optional[float],
    ) -> str:
        """Create deterministic key for observation deduplication."""
        # 1-second bucket for PTS
        bucket = int(pts_ms / 1000.0) if pts_ms is not None else 0
        return f"{camera_id}:{track_id}:{norm_reg}:{bucket}"

    def _load_from_disk(self) -> None:
        """Load existing data from disk into memory indices."""
        # 1. Load observations.jsonl
        if self.observations_file.exists():
            try:
                with open(self.observations_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        obs = VehicleObservation.from_dict(data)
                        self._observations.append(obs)
                        pts = obs.recognition_pts_ms if obs.recognition_pts_ms is not None else obs.first_seen_pts_ms
                        key = self._make_obs_key(
                            obs.camera_id, obs.track_id,
                            obs.normalized_registration_number, pts
                        )
                        self._observation_keys.add(key)
            except Exception as e:
                logger.error("Failed to load observations from disk: %s", e)

        # 2. Load vehicles.json
        if self.vehicles_file.exists():
            try:
                with open(self.vehicles_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        veh = ObservedVehicle.from_dict(item)
                        self._vehicles[veh.normalized_registration_number] = veh
            except Exception as e:
                logger.error("Failed to load vehicles from disk: %s", e)

        # 3. Load tracks.json
        if self.tracks_file.exists():
            try:
                with open(self.tracks_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        trk = VehicleTrackRecord.from_dict(item)
                        self._tracks[f"{trk.camera_id}_{trk.track_id}"] = trk
            except Exception as e:
                logger.error("Failed to load tracks from disk: %s", e)

        # 4. Load uncertain_observations.jsonl
        if self.uncertain_file.exists():
            try:
                with open(self.uncertain_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        u = UncertainObservation.from_dict(data)
                        self._uncertain.append(u)
                        self._uncertain_keys.add(f"{u.camera_id}:{u.track_id}:{u.raw_text}")
            except Exception as e:
                logger.error("Failed to load uncertain observations from disk: %s", e)

    def add_observation(self, observation: VehicleObservation) -> bool:
        """Add an observation if not duplicate. Flushes to observations.jsonl."""
        pts = observation.recognition_pts_ms if observation.recognition_pts_ms is not None else observation.first_seen_pts_ms
        key = self._make_obs_key(
            observation.camera_id,
            observation.track_id,
            observation.normalized_registration_number,
            pts,
        )
        if key in self._observation_keys:
            return False  # Idempotent skip

        self._observations.append(observation)
        self._observation_keys.add(key)

        with open(self.observations_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(observation.to_dict()) + "\n")
        return True

    def add_uncertain_observation(self, observation: UncertainObservation) -> bool:
        """Add an uncertain/unreadable observation if not duplicate."""
        key = f"{observation.camera_id}:{observation.track_id}:{observation.raw_text}"
        if key in self._uncertain_keys:
            return False

        self._uncertain.append(observation)
        self._uncertain_keys.add(key)

        with open(self.uncertain_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(observation.to_dict()) + "\n")
        return True

    def save_vehicle(self, vehicle: ObservedVehicle, flush: bool = True) -> None:
        """Save or update an aggregated vehicle record in memory and sync vehicles.json."""
        self._vehicles[vehicle.normalized_registration_number] = vehicle
        if flush:
            self._flush_vehicles()

    def _flush_vehicles(self) -> None:
        """Write all vehicles to vehicles.json atomically."""
        tmp_file = self.storage_dir / "vehicles.json.tmp"
        payload = [v.to_dict() for v in self._vehicles.values()]
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        _safe_replace(tmp_file, self.vehicles_file)

    def save_track(self, track: VehicleTrackRecord, flush: bool = True) -> None:
        """Save or update a track record in memory and sync tracks.json."""
        self._tracks[f"{track.camera_id}_{track.track_id}"] = track
        if flush:
            self._flush_tracks()

    def _flush_tracks(self) -> None:
        """Write all tracks to tracks.json atomically."""
        tmp_file = self.storage_dir / "tracks.json.tmp"
        payload = [t.to_dict() for t in self._tracks.values()]
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        _safe_replace(tmp_file, self.tracks_file)

    def flush(self) -> None:
        """Flush both vehicles and tracks to disk."""
        self._flush_tracks()
        self._flush_vehicles()

    def get_vehicle(self, normalized_registration: str) -> Optional[ObservedVehicle]:
        return self._vehicles.get(normalized_registration)

    def get_vehicle_observations(self, normalized_registration: str) -> List[VehicleObservation]:
        return [
            o for o in self._observations
            if o.normalized_registration_number == normalized_registration
        ]

    def get_vehicle_cameras(self, normalized_registration: str) -> List[str]:
        veh = self.get_vehicle(normalized_registration)
        return veh.cameras if veh else []

    def get_vehicle_timeline(self, normalized_registration: str) -> List[Dict[str, Any]]:
        veh = self.get_vehicle(normalized_registration)
        if not veh:
            return []
        return [entry.to_dict() for entry in veh.timeline]

    def get_track(self, track_id: str, camera_id: Optional[str] = None) -> Optional[VehicleTrackRecord]:
        if camera_id:
            return self._tracks.get(f"{camera_id}_{track_id}")
        for k, trk in self._tracks.items():
            if trk.track_id == track_id:
                return trk
        return None

    def list_vehicles(self, limit: int = 100, offset: int = 0) -> List[ObservedVehicle]:
        all_v = sorted(self._vehicles.values(), key=lambda v: v.observation_count, reverse=True)
        return all_v[offset : offset + limit]

    def count_observations(self) -> int:
        return len(self._observations)

    def get_statistics(self) -> Dict[str, Any]:
        """Compute network-wide observation statistics."""
        obs_by_cam: Dict[str, int] = {}
        obs_by_status: Dict[str, int] = {}
        total_consensus = 0.0

        for o in self._observations:
            obs_by_cam[o.camera_id] = obs_by_cam.get(o.camera_id, 0) + 1
            obs_by_status[o.status] = obs_by_status.get(o.status, 0) + 1
            total_consensus += o.consensus_score

        avg_consensus = (total_consensus / float(len(self._observations))) if self._observations else 0.0

        # Top observed vehicles
        top_vehicles = [
            {
                "registration_number": v.registration_number,
                "observation_count": v.observation_count,
                "camera_count": v.camera_count,
                "cameras": v.cameras,
                "best_consensus_score": v.best_consensus_score,
            }
            for v in self.list_vehicles(limit=5)
        ]

        return {
            "total_observed_vehicles": len(self._vehicles),
            "total_observations": len(self._observations),
            "total_tracks": len(self._tracks),
            "uncertain_observations_count": len(self._uncertain),
            "average_consensus_score": round(avg_consensus, 4),
            "observations_by_camera": obs_by_cam,
            "observations_by_status": obs_by_status,
            "top_observed_vehicles": top_vehicles,
        }
