"""Event Schemas and Envelope Models for Step 10 Kafka Event Pipeline.

Every message flowing across the platform adheres to a common envelope:
  - event_id: Unique UUID string (e.g. EVT-550e8400-e29b-41d4-a716-446655440000)
  - event_type: Event classification string (e.g. vehicle.anpr, watchlist.match)
  - schema_version: Version identifier (default: '1.0')
  - created_at: ISO 8601 UTC timestamp
  - source: Originating system and camera metadata
  - correlation_id: Distributed tracing identifier across the CCTV pipeline
  - payload: Domain-specific payload dictionary
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

CURRENT_SCHEMA_VERSION = "1.0"


def _get_utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_event_id(prefix: str = "EVT") -> str:
    """Generate a globally unique event ID with UUIDv4."""
    return f"{prefix}-{uuid.uuid4()}"


@dataclass
class EventSource:
    """Metadata describing the origin of an event."""
    system: str  # e.g. 'cctv-ai', 'vehicle-detector', 'watchlist-matcher'
    camera_id: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EventSource:
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)


@dataclass
class EventEnvelope:
    """Common envelope wrapping all Kafka messages."""
    event_id: str
    event_type: str
    schema_version: str = CURRENT_SCHEMA_VERSION
    created_at: str = field(default_factory=_get_utc_now_iso)
    source: EventSource = field(default_factory=lambda: EventSource(system="cctv-ai"))
    correlation_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["source"] = self.source.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EventEnvelope:
        copied = dict(data)
        src_raw = copied.get("source") or {}
        source_obj = EventSource.from_dict(src_raw) if isinstance(src_raw, dict) else EventSource(system="unknown")
        copied["source"] = source_obj
        filtered = {k: v for k, v in copied.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)


# ════════════════════════════════════════════════════════════════════════════
# Domain Event Builders
# ════════════════════════════════════════════════════════════════════════════

def build_vehicle_detection_event(
    camera_id: str,
    track_id: str,
    vehicle_class: str,
    confidence: float,
    bbox: List[int],
    pts_ms: Optional[float],
    event_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> EventEnvelope:
    """Construct a vehicle.detection event envelope."""
    payload = {
        "camera_id": camera_id,
        "track_id": track_id,
        "vehicle_class": vehicle_class,
        "confidence": round(confidence, 4),
        "bbox": bbox,
        "pts_ms": pts_ms,
    }
    return EventEnvelope(
        event_id=event_id or generate_event_id(),
        event_type="vehicle.detection",
        schema_version=CURRENT_SCHEMA_VERSION,
        source=EventSource(system="vehicle-detector", camera_id=camera_id),
        correlation_id=correlation_id or f"TRK-{camera_id}-{track_id}",
        payload=payload,
    )


def build_anpr_event(
    camera_id: str,
    track_id: str,
    registration_number: str,
    normalized_registration_number: str,
    recognition_status: str,
    consensus_score: float,
    ocr_confidence: float,
    plate_detection_confidence: float,
    plate_quality_score: float,
    recognition_pts_ms: Optional[float],
    evidence_image: Optional[str] = None,
    source_time: Optional[str] = None,
    source_time_status: str = "NOT_RESOLVED",
    event_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> EventEnvelope:
    """Construct a vehicle.anpr event envelope."""
    payload = {
        "camera_id": camera_id,
        "track_id": track_id,
        "registration_number": registration_number,
        "normalized_registration_number": normalized_registration_number,
        "recognition_status": recognition_status,
        "consensus_score": round(consensus_score, 4),
        "ocr_confidence": round(ocr_confidence, 4),
        "plate_detection_confidence": round(plate_detection_confidence, 4),
        "plate_quality_score": round(plate_quality_score, 4),
        "recognition_pts_ms": recognition_pts_ms,
        "evidence_image": evidence_image,
        "source_time": source_time,
        "source_time_status": source_time_status,
    }
    return EventEnvelope(
        event_id=event_id or generate_event_id(),
        event_type="vehicle.anpr",
        schema_version=CURRENT_SCHEMA_VERSION,
        source=EventSource(system="cctv-ai", camera_id=camera_id),
        correlation_id=correlation_id or f"TRK-{camera_id}-{track_id}",
        payload=payload,
    )


def build_watchlist_match_event(
    match_id: str,
    observation_id: str,
    watchlist_id: Optional[str],
    decision: str,
    registration_number: str,
    normalized_registration_number: str,
    camera_id: str,
    track_id: str,
    consensus_score: float,
    match_score: float,
    watchlist_metadata: Optional[Dict[str, Any]],
    evidence_image: Optional[str],
    recognition_pts_ms: Optional[float],
    is_deduplicated: bool = False,
    event_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> EventEnvelope:
    """Construct a watchlist.matches event envelope."""
    payload = {
        "match_id": match_id,
        "observation_id": observation_id,
        "watchlist_id": watchlist_id,
        "decision": decision,
        "registration_number": registration_number,
        "normalized_registration_number": normalized_registration_number,
        "camera_id": camera_id,
        "track_id": track_id,
        "consensus_score": round(consensus_score, 4),
        "match_score": round(match_score, 4),
        "watchlist_metadata": watchlist_metadata,
        "evidence_image": evidence_image,
        "recognition_pts_ms": recognition_pts_ms,
        "is_deduplicated": is_deduplicated,
    }
    return EventEnvelope(
        event_id=event_id or generate_event_id(),
        event_type="watchlist.match",
        schema_version=CURRENT_SCHEMA_VERSION,
        source=EventSource(system="watchlist-matcher", camera_id=camera_id),
        correlation_id=correlation_id or f"TRK-{camera_id}-{track_id}",
        payload=payload,
    )


def build_alert_event(
    match_id: str,
    watchlist_id: str,
    registration_number: str,
    priority: str,
    category: str,
    decision: str,
    camera_id: str,
    track_id: str,
    recognition_pts_ms: Optional[float],
    evidence_image: Optional[str],
    recognition_consensus: float,
    event_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> EventEnvelope:
    """Construct an alert-ready event envelope for the alerts topic."""
    payload = {
        "match_id": match_id,
        "watchlist_id": watchlist_id,
        "registration_number": registration_number,
        "priority": priority,
        "category": category,
        "decision": decision,
        "camera_id": camera_id,
        "track_id": track_id,
        "recognition_pts_ms": recognition_pts_ms,
        "evidence_image": evidence_image,
        "recognition_consensus": round(recognition_consensus, 4),
    }
    return EventEnvelope(
        event_id=event_id or generate_event_id(),
        event_type="watchlist.alert",
        schema_version=CURRENT_SCHEMA_VERSION,
        source=EventSource(system="watchlist-matcher", camera_id=camera_id),
        correlation_id=correlation_id or f"TRK-{camera_id}-{track_id}",
        payload=payload,
    )


def build_dlq_event(
    original_event_id: str,
    original_topic: str,
    error_code: str,
    error_message: str,
    retry_count: int,
    raw_payload: Optional[str] = None,
    event_id: Optional[str] = None,
) -> EventEnvelope:
    """Construct a dead-letter queue event envelope."""
    payload = {
        "original_event_id": original_event_id,
        "original_topic": original_topic,
        "error_code": error_code,
        "error_message": error_message,
        "retry_count": retry_count,
        "raw_payload": raw_payload,
        "failed_at_utc": _get_utc_now_iso(),
    }
    return EventEnvelope(
        event_id=event_id or generate_event_id(prefix="DLQ"),
        event_type="vehicle.events.dlq",
        schema_version=CURRENT_SCHEMA_VERSION,
        source=EventSource(system="kafka-pipeline-dlq"),
        payload=payload,
    )
