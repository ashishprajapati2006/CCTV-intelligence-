"""Pydantic schemas for the CCTV Intelligence Platform API."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CameraSchema(BaseModel):
    camera_id: str
    name: str
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    status: str = "active"
    codec: str = "H264"
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    bitrate: Optional[int] = None
    rtsp_url: Optional[str] = None
    hls_url: Optional[str] = None
    webrtc_url: Optional[str] = None
    department: str = "Gujarat Police"
    is_spatial: bool = False
    camera_type: Optional[str] = "Fixed"
    ai_capabilities: Optional[List[str]] = Field(default_factory=list)
    alert_count: int = 0
    observation_count: int = 0


class CameraHealthSchema(BaseModel):
    camera_id: str
    connected: bool = False
    frames_received: int = 0
    last_pts_ms: Optional[float] = None
    last_receive_time: Optional[float] = None
    reconnect_count: int = 0
    decode_errors: int = 0
    pts_discontinuities: int = 0
    width: Optional[int] = None
    height: Optional[int] = None
    codec: Optional[str] = None
    status_label: str = "ONLINE"


class StreamDescriptorSchema(BaseModel):
    camera_id: str
    protocol: str  # "hls" or "webrtc"
    stream_url: str
    stream_type: str = "live"
    requires_proxy: bool = False


class VehicleObservationSchema(BaseModel):
    observation_id: str
    camera_id: str
    track_id: str
    registration_number: str
    normalized_registration_number: str
    recognition_status: str
    consensus_score: float
    ocr_confidence: float
    first_seen_pts_ms: Optional[float] = None
    recognition_pts_ms: Optional[float] = None
    last_seen_pts_ms: Optional[float] = None
    source_time: Optional[str] = None
    source_time_status: str = "NOT_RESOLVED"
    evidence_image_path: Optional[str] = None
    evidence_url: Optional[str] = None
    ingested_at_utc: str


class ObservedVehicleSchema(BaseModel):
    vehicle_id: str
    registration_number: str
    normalized_registration_number: str
    camera_count: int = 1
    observation_count: int = 1
    track_count: int = 1
    best_consensus_score: float = 0.0
    average_consensus_score: float = 0.0
    status: str = "OBSERVED"
    cameras: List[str] = []
    first_seen: Optional[Dict[str, Any]] = None
    last_seen: Optional[Dict[str, Any]] = None
    timeline: List[Dict[str, Any]] = []
    ingested_at_utc: Optional[str] = None
    updated_at_utc: Optional[str] = None


class WatchlistEntrySchema(BaseModel):
    watchlist_id: str
    registration_number: str
    normalized_registration_number: str
    category: str
    priority: str
    reason: Optional[str] = None
    status: str = "ACTIVE"
    synthetic: bool = True
    source: str = "SYNTHETIC_DEMO"
    created_at_utc: Optional[str] = None


class AlertSchema(BaseModel):
    alert_id: str
    match_id: str
    observation_id: str
    registration_number: str
    normalized_registration_number: str
    watchlist_id: Optional[str] = ""
    category: str
    priority: str
    decision: str
    camera_id: str
    track_id: str
    status: str = "NEW"  # NEW, ACKNOWLEDGED, UNDER_REVIEW, ESCALATED, RESOLVED, DISMISSED
    recognition_confidence: float
    consensus_score: float
    first_seen_pts_ms: Optional[float] = None
    recognition_pts_ms: Optional[float] = None
    source_time: Optional[str] = None
    source_time_status: str = "NOT_RESOLVED"
    evidence_image: Optional[str] = None
    evidence_url: Optional[str] = None
    matched_at_utc: str
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None
    operator_notes: Optional[str] = None
    audit_history: List[Dict[str, Any]] = []


class AlertActionRequest(BaseModel):
    action: str  # acknowledge, escalate, resolve, dismiss
    operator: str = "Control Room Operator"
    notes: Optional[str] = None


class DashboardStatsSchema(BaseModel):
    total_cameras: int = 30
    online_cameras: int = 30
    degraded_cameras: int = 0
    offline_cameras: int = 0
    ai_active_cameras: int = 6
    total_observed_vehicles: int = 0
    total_sightings: int = 0
    active_watchlist_targets: int = 0
    active_alerts: int = 0
    critical_alerts: int = 0
    recent_anpr_observations: List[Dict[str, Any]] = []
    recent_alerts: List[Dict[str, Any]] = []
