"""Domain models for Step 8 (Watchlist) and Step 9 (Real-time Watchlist Matching).

All records represent demonstration data for the Gujarat Police Innovation Hackathon 2026.
Every record is explicitly tagged with source='SYNTHETIC_DEMO' and synthetic=True.
Categories are prefixed with 'DEMO_' to prevent confusion with real police records.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

ALLOWED_VEHICLE_CATEGORIES = [
    "DEMO_STOLEN_VEHICLE",
    "DEMO_BLACKLISTED_VEHICLE",
    "DEMO_VEHICLE_OF_INTEREST",
    "DEMO_MISSING_VEHICLE",
]

ALLOWED_PRIORITIES = [
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
]

ALLOWED_STATUSES = [
    "ACTIVE",
    "INACTIVE",
    "EXPIRED",
]

ALLOWED_MATCH_DECISIONS = [
    "MATCH",
    "NO_MATCH",
    "POSSIBLE_MATCH_REVIEW",
    "NOT_ELIGIBLE",
]


def _get_utc_now_iso() -> str:
    """Generate ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class WatchlistVehicle:
    """Vehicle watchlist record.

    Maps to future PostgreSQL table: watchlist_vehicles
    """
    watchlist_id: str
    registration_number: str
    normalized_registration_number: str
    category: str = "DEMO_VEHICLE_OF_INTEREST"
    priority: str = "HIGH"
    status: str = "ACTIVE"
    entity_type: str = "vehicle"
    source: str = "SYNTHETIC_DEMO"
    synthetic: bool = True
    description: str = "Synthetic demonstration record"
    notes: Optional[str] = None
    created_at: str = field(default_factory=_get_utc_now_iso)
    updated_at: str = field(default_factory=_get_utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WatchlistVehicle:
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)


@dataclass
class WatchlistPerson:
    """Person of interest watchlist record (Schema preview only).

    Maps to future PostgreSQL table: watchlist_persons
    Contains NO real personal identifiable information (PII).
    """
    watchlist_id: str
    entity_type: str = "person"
    category: str = "DEMO_PERSON_OF_INTEREST"
    priority: str = "HIGH"
    status: str = "ACTIVE"
    source: str = "SYNTHETIC_DEMO"
    synthetic: bool = True
    description: str = "Synthetic demonstration record (No real PII)"
    created_at: str = field(default_factory=_get_utc_now_iso)
    updated_at: str = field(default_factory=_get_utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WatchlistPerson:
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)


@dataclass
class WatchlistMetadataSnapshot:
    """Immutable snapshot of watchlist record metadata taken at match decision time.

    Preserves auditability even if watchlist records are subsequently altered or deactivated.
    """
    watchlist_id: str
    category: str
    priority: str
    status: str
    source: str = "SYNTHETIC_DEMO"
    synthetic: bool = True
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WatchlistMetadataSnapshot:
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)

    @classmethod
    def from_watchlist_vehicle(cls, v: WatchlistVehicle) -> WatchlistMetadataSnapshot:
        return cls(
            watchlist_id=v.watchlist_id,
            category=v.category,
            priority=v.priority,
            status=v.status,
            source=v.source,
            synthetic=v.synthetic,
            description=v.description,
        )


@dataclass
class MatchDecision:
    """Auditable match decision record produced by Step 9 Watchlist Matching Engine.

    Maps to future PostgreSQL table: watchlist_matches
    Represents an alert-ready event for Step 10 Kafka transport.
    """
    match_id: str
    observation_id: str
    decision: str  # "MATCH", "NO_MATCH", "POSSIBLE_MATCH_REVIEW", "NOT_ELIGIBLE"
    decision_reason: str
    registration_number: Optional[str] = None
    normalized_registration_number: Optional[str] = None
    watchlist_id: Optional[str] = None

    # Confidence and score separation (Critical: OCR conf != Match score)
    watchlist_match_score: float = 0.0  # 1.0 for exact match, [0.0, 1.0] for fuzzy, 0.0 for none
    recognition_confidence: float = 0.0  # Raw OCR confidence
    consensus_score: float = 0.0        # Step 6 ANPR consensus score
    plate_detection_confidence: float = 0.0
    plate_quality_score: float = 0.0

    # Camera & Track Provenance
    camera_id: str = "unknown"
    track_id: str = "unknown"
    first_seen_pts_ms: Optional[float] = None
    recognition_pts_ms: Optional[float] = None
    last_seen_pts_ms: Optional[float] = None
    source_time: Optional[str] = None
    source_time_status: str = "NOT_RESOLVED"

    # Evidence Reference
    evidence_image: Optional[str] = None
    evidence_filename_pts_status: str = "UNKNOWN"

    # Watchlist Metadata Snapshot
    watchlist_metadata: Optional[WatchlistMetadataSnapshot] = None

    # Deduplication & Track Aggregation
    is_deduplicated: bool = False
    supporting_observations_count: int = 1

    # Optional Fuzzy Match Diagnostics
    fuzzy_diff: Optional[Dict[str, Any]] = None

    # Timing
    matched_at_utc: str = field(default_factory=_get_utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.watchlist_metadata:
            d["watchlist_metadata"] = self.watchlist_metadata.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MatchDecision:
        copied = dict(data)
        if copied.get("watchlist_metadata") and isinstance(copied["watchlist_metadata"], dict):
            copied["watchlist_metadata"] = WatchlistMetadataSnapshot.from_dict(copied["watchlist_metadata"])
        filtered = {k: v for k, v in copied.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)
