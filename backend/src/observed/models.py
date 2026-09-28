"""Domain models for the Observed Vehicle Database (Step 7).

Represents vehicles actually observed by the CCTV network.
Enforces strict timestamp semantics:
  1. Media PTS (first_seen_pts_ms, recognition_pts_ms, last_seen_pts_ms)
  2. Source/calendar time (source_time, source_time_status: NOT_RESOLVED | RESOLVED)
  3. Ingestion time (ingested_at_utc) - current application time only.
Maintains strict separation from watchlists (no 'wanted' or 'suspicious' flags).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _get_utc_now_iso() -> str:
    """Generate ISO 8601 UTC timestamp for application ingestion operations only."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class VehicleObservation:
    """An individual plate observation event from a camera stream.

    Maps to future PostgreSQL table: vehicle_observations
    """
    observation_id: str
    camera_id: str
    track_id: str
    registration_number: str
    normalized_registration_number: str
    first_seen_pts_ms: Optional[float]
    recognition_pts_ms: Optional[float]
    last_seen_pts_ms: Optional[float]
    source_time: Optional[str] = None
    source_time_status: str = "NOT_RESOLVED"  # NOT_RESOLVED | RESOLVED
    ocr_confidence: float = 0.0
    plate_detection_confidence: float = 0.0
    plate_quality_score: float = 0.0
    consensus_score: float = 0.0
    supporting_frame_count: int = 1
    status: str = "CONFIRMED"  # CONFIRMED, PROBABLE, UNCERTAIN, UNREADABLE
    evidence_image: Optional[str] = None
    evidence_filename_pts_status: str = "UNKNOWN"  # MATCH, DISCREPANCY, UNKNOWN
    evidence_filename_pts_ms: Optional[float] = None
    source: str = "sentinel"
    loop_instance: Optional[int] = None
    ingested_at_utc: str = field(default_factory=_get_utc_now_iso)

    @property
    def observed_at_pts_ms(self) -> Optional[float]:
        """Deprecated alias for first_seen_pts_ms for backward compatibility."""
        return self.first_seen_pts_ms

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Include backward-compatible alias for any legacy consumers
        d["observed_at_pts_ms"] = self.first_seen_pts_ms
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VehicleObservation:
        data_copy = dict(data)
        # Backward compatibility for old records
        if "first_seen_pts_ms" not in data_copy and "observed_at_pts_ms" in data_copy:
            data_copy["first_seen_pts_ms"] = data_copy["observed_at_pts_ms"]
        if "recognition_pts_ms" not in data_copy:
            data_copy["recognition_pts_ms"] = data_copy.get("first_seen_pts_ms")
        if "last_seen_pts_ms" not in data_copy:
            data_copy["last_seen_pts_ms"] = data_copy.get("first_seen_pts_ms")
        if "ingested_at_utc" not in data_copy and "created_at" in data_copy:
            data_copy["ingested_at_utc"] = data_copy["created_at"]
        if "source_time_status" not in data_copy:
            data_copy["source_time_status"] = "NOT_RESOLVED"
        filtered = {k: v for k, v in data_copy.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)


@dataclass
class SightingRecord:
    """Metadata for first_seen or last_seen vehicle sighting."""
    camera_id: str
    pts_ms: Optional[float]
    source_time: Optional[str] = None
    source_time_status: str = "NOT_RESOLVED"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TimelineEntry:
    """Observation entry in a vehicle's cross-camera observation history."""
    camera_id: str
    track_id: str
    first_seen_pts_ms: Optional[float]
    recognition_pts_ms: Optional[float]
    last_seen_pts_ms: Optional[float]
    source_time: Optional[str] = None
    source_time_status: str = "NOT_RESOLVED"
    status: str = "CONFIRMED"
    consensus_score: float = 0.0
    evidence_image: Optional[str] = None
    evidence_filename_pts_status: str = "UNKNOWN"
    ingested_at_utc: str = field(default_factory=_get_utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ObservedVehicle:
    """Aggregated vehicle-level identity record.

    One record per normalized registration number.
    Contains NO watchlist, wanted, or suspicious fields.
    Maps to future PostgreSQL table: observed_vehicles
    """
    vehicle_id: str
    registration_number: str
    normalized_registration_number: str
    first_seen: SightingRecord
    last_seen: SightingRecord
    camera_count: int
    cameras: List[str]
    observation_count: int
    track_count: int
    best_consensus_score: float
    average_consensus_score: float
    status: str = "OBSERVED"
    timeline: List[TimelineEntry] = field(default_factory=list)
    ingested_at_utc: str = field(default_factory=_get_utc_now_iso)
    updated_at_utc: str = field(default_factory=_get_utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ObservedVehicle:
        data_copy = dict(data)
        if isinstance(data_copy.get("first_seen"), dict):
            fs = data_copy["first_seen"]
            data_copy["first_seen"] = SightingRecord(
                camera_id=fs.get("camera_id", "unknown"),
                pts_ms=fs.get("pts_ms"),
                source_time=fs.get("source_time"),
                source_time_status=fs.get("source_time_status", "NOT_RESOLVED"),
            )
        if isinstance(data_copy.get("last_seen"), dict):
            ls = data_copy["last_seen"]
            data_copy["last_seen"] = SightingRecord(
                camera_id=ls.get("camera_id", "unknown"),
                pts_ms=ls.get("pts_ms"),
                source_time=ls.get("source_time"),
                source_time_status=ls.get("source_time_status", "NOT_RESOLVED"),
            )
        if "timeline" in data_copy and isinstance(data_copy["timeline"], list):
            parsed_timeline = []
            for item in data_copy["timeline"]:
                if isinstance(item, dict):
                    # Handle backward compatibility
                    entry_copy = dict(item)
                    if "first_seen_pts_ms" not in entry_copy and "pts_ms" in entry_copy:
                        entry_copy["first_seen_pts_ms"] = entry_copy["pts_ms"]
                    if "recognition_pts_ms" not in entry_copy:
                        entry_copy["recognition_pts_ms"] = entry_copy.get("first_seen_pts_ms")
                    if "last_seen_pts_ms" not in entry_copy:
                        entry_copy["last_seen_pts_ms"] = entry_copy.get("first_seen_pts_ms")
                    filtered_entry = {k: v for k, v in entry_copy.items() if k in TimelineEntry.__dataclass_fields__}
                    parsed_timeline.append(TimelineEntry(**filtered_entry))
                else:
                    parsed_timeline.append(item)
            data_copy["timeline"] = parsed_timeline
        filtered = {k: v for k, v in data_copy.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)


@dataclass
class SupportingFrameRef:
    """Reference to supporting frame in a vehicle track."""
    frame_id: int
    pts_ms: Optional[float]
    evidence_image: Optional[str] = None
    filename_pts_ms: Optional[float] = None
    filename_pts_status: str = "UNKNOWN"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VehicleTrackRecord:
    """Track-level evidence and association information.

    Maps to future PostgreSQL table: vehicle_tracks
    """
    track_id: str
    camera_id: str
    registration_number: Optional[str]
    normalized_registration_number: Optional[str]
    first_seen_pts_ms: Optional[float]
    recognition_pts_ms: Optional[float]
    last_seen_pts_ms: Optional[float]
    source_time: Optional[str] = None
    source_time_status: str = "NOT_RESOLVED"
    supporting_frames: List[SupportingFrameRef] = field(default_factory=list)
    consensus_score: float = 0.0
    status: str = "CONFIRMED"  # CONFIRMED, PROBABLE, UNCERTAIN, UNREADABLE
    observation_count: int = 1
    loop_instance: Optional[int] = None
    ingested_at_utc: str = field(default_factory=_get_utc_now_iso)

    # Backward compatibility alias
    @property
    def first_pts_ms(self) -> Optional[float]:
        return self.first_seen_pts_ms

    @property
    def last_pts_ms(self) -> Optional[float]:
        return self.last_seen_pts_ms

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["first_pts_ms"] = self.first_seen_pts_ms
        d["last_pts_ms"] = self.last_seen_pts_ms
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VehicleTrackRecord:
        data_copy = dict(data)
        if "first_seen_pts_ms" not in data_copy and "first_pts_ms" in data_copy:
            data_copy["first_seen_pts_ms"] = data_copy["first_pts_ms"]
        if "last_seen_pts_ms" not in data_copy and "last_pts_ms" in data_copy:
            data_copy["last_seen_pts_ms"] = data_copy["last_pts_ms"]
        if "recognition_pts_ms" not in data_copy:
            data_copy["recognition_pts_ms"] = data_copy.get("first_seen_pts_ms")
        if "supporting_frames" in data_copy and isinstance(data_copy["supporting_frames"], list):
            data_copy["supporting_frames"] = [
                SupportingFrameRef(**item) if isinstance(item, dict) else item
                for item in data_copy["supporting_frames"]
            ]
        filtered = {k: v for k, v in data_copy.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)


@dataclass
class UncertainObservation:
    """Audit record for low-confidence or unreadable recognition evidence.

    Preserved for audit/provenance without polluting definitive vehicle identities.
    Maps to future PostgreSQL table: uncertain_observations
    """
    observation_id: str
    camera_id: str
    track_id: str
    raw_text: str
    normalized_text: str
    status: str  # UNCERTAIN, UNREADABLE
    reason: str
    evidence_image: Optional[str] = None
    first_seen_pts_ms: Optional[float] = None
    recognition_pts_ms: Optional[float] = None
    last_seen_pts_ms: Optional[float] = None
    source_time: Optional[str] = None
    source_time_status: str = "NOT_RESOLVED"
    loop_instance: Optional[int] = None
    ingested_at_utc: str = field(default_factory=_get_utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UncertainObservation:
        data_copy = dict(data)
        if "first_seen_pts_ms" not in data_copy and "pts_ms" in data_copy:
            data_copy["first_seen_pts_ms"] = data_copy["pts_ms"]
        filtered = {k: v for k, v in data_copy.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)
