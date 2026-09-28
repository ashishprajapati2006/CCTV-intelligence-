"""Domain Models for Step 11 — Cross-Camera Correlation & Vehicle Journey Reconstruction.

Preserves camera-local media PTS independently from source calendar time and
application ingestion time. Enforces the Common-Clock Rule.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class PlausibilityState(str, Enum):
    PLAUSIBLE = "PLAUSIBLE"
    POSSIBLE = "POSSIBLE"
    ANOMALOUS = "ANOMALOUS"
    UNKNOWN = "UNKNOWN"


class OrderingMode(str, Enum):
    SOURCE_TIME = "SOURCE_TIME"
    CAMERA_LOCAL = "CAMERA_LOCAL"


def _get_utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CrossCameraObservation:
    """Canonical single observation of a recognized vehicle from Step 7."""
    observation_id: str
    vehicle_id: str
    registration_number: str
    normalized_registration_number: str
    camera_id: str
    track_id: str
    first_seen_pts_ms: Optional[float] = None
    recognition_pts_ms: Optional[float] = None
    last_seen_pts_ms: Optional[float] = None
    source_time: Optional[str] = None
    source_time_status: str = "NOT_RESOLVED"
    loop_instance: Optional[int] = None
    recognition_status: str = "CONFIRMED"
    consensus_score: float = 0.0
    ocr_confidence: float = 0.0
    camera_name: Optional[str] = None
    camera_latitude: Optional[float] = None
    camera_longitude: Optional[float] = None
    evidence_image: Optional[str] = None
    evidence_filename_pts_status: str = "UNKNOWN"
    source: str = "sentinel"
    ingested_at_utc: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CrossCameraObservation:
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)


@dataclass
class CameraObservationSegment:
    """Atomic unit for cross-camera correlation representing an aggregated track on one camera."""
    segment_id: str
    vehicle_id: str
    registration_number: str
    normalized_registration_number: str
    camera_id: str
    track_id: str
    first_seen_pts_ms: Optional[float] = None
    recognition_pts_ms: Optional[float] = None
    last_seen_pts_ms: Optional[float] = None
    source_time: Optional[str] = None
    source_time_status: str = "NOT_RESOLVED"
    loop_instance: Optional[int] = None
    recognition_status: str = "CONFIRMED"
    consensus_score: float = 0.0
    ocr_confidence: float = 0.0
    camera_name: Optional[str] = None
    camera_latitude: Optional[float] = None
    camera_longitude: Optional[float] = None
    evidence_image: Optional[str] = None
    evidence_filename_pts_status: str = "UNKNOWN"
    supporting_observation_count: int = 1
    observation_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CameraObservationSegment:
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)


@dataclass
class JourneyLeg:
    """Directed connection between two consecutive observation segments."""
    leg_id: str
    vehicle_id: str
    from_observation_id: str
    to_observation_id: str
    from_camera_id: str
    to_camera_id: str
    from_camera_name: Optional[str] = None
    to_camera_name: Optional[str] = None
    # Source time passthrough (Parts 5, 9)
    from_source_time: Optional[str] = None
    to_source_time: Optional[str] = None
    time_delta_seconds: Optional[float] = None
    # Spatial fields (Parts 8, 9, 10)
    straight_line_distance_m: Optional[float] = None
    distance_status: str = "UNAVAILABLE"        # "AVAILABLE" | "UNAVAILABLE"
    distance_label: str = "STRAIGHT_LINE_DISTANCE"  # Never "road distance"
    implied_speed_kmh: Optional[float] = None
    implied_speed_label: str = "IMPLIED STRAIGHT-LINE SPEED"
    source_time_status: str = "NOT_RESOLVED"
    plausibility: PlausibilityState = PlausibilityState.UNKNOWN
    plausibility_reason: str = "Time basis unresolved or coordinates missing"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["plausibility"] = self.plausibility.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> JourneyLeg:
        copied = dict(data)
        if "plausibility" in copied:
            copied["plausibility"] = PlausibilityState(copied["plausibility"])
        filtered = {k: v for k, v in copied.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)


@dataclass
class JourneyConfidenceScore:
    """Engineering confidence indicator assessing data completeness and plausibility."""
    overall_score: float
    recognition_quality: float
    temporal_resolution: float
    spatial_resolution: float
    plausibility_consistency: float
    notes: List[str] = field(default_factory=list)
    # Structured explanations per confidence component (Part 21)
    explanations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> JourneyConfidenceScore:
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)


@dataclass
class VehicleJourney:
    """Complete cross-camera vehicle journey / observation sequence model."""
    journey_id: str
    vehicle_id: str
    registration_number: str
    normalized_registration_number: str
    ordering_mode: str  # "SOURCE_TIME" or "CAMERA_LOCAL"
    status: str        # "RECONSTRUCTED" or "OBSERVATION_SEQUENCE_ONLY"
    segments: List[CameraObservationSegment] = field(default_factory=list)
    legs: List[JourneyLeg] = field(default_factory=list)
    confidence_score: Optional[JourneyConfidenceScore] = None
    total_distance_m: Optional[float] = None
    total_duration_seconds: Optional[float] = None
    # Part 24 — machine-readable journey schema fields
    time_resolution: str = "NOT_RESOLVED"    # "NOT_RESOLVED" | "PARTIAL" | "RESOLVED"
    spatial_resolution: str = "UNAVAILABLE"  # "UNAVAILABLE" | "PARTIAL" | "AVAILABLE"
    limitations: List[str] = field(default_factory=list)
    media_session_id: Optional[str] = None
    created_at: str = field(default_factory=_get_utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["segments"] = [s.to_dict() for s in self.segments]
        d["legs"] = [leg.to_dict() for leg in self.legs]
        if self.confidence_score:
            d["confidence_score"] = self.confidence_score.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VehicleJourney:
        copied = dict(data)
        if "segments" in copied:
            copied["segments"] = [CameraObservationSegment.from_dict(s) for s in copied["segments"]]
        if "legs" in copied:
            copied["legs"] = [JourneyLeg.from_dict(leg) for leg in copied["legs"]]
        if "confidence_score" in copied and copied["confidence_score"]:
            copied["confidence_score"] = JourneyConfidenceScore.from_dict(copied["confidence_score"])
        filtered = {k: v for k, v in copied.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)
