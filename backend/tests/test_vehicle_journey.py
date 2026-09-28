"""Comprehensive Unit Tests for Step 11 — Vehicle Journey Reconstruction.

Tests the 7 scenarios from Part 26:
  Scenario A: Chronologically resolved source time (ordered journey + legs)
  Scenario B: Camera-local PTS only (observation sequence only, no cross-camera clock)
  Scenario C: Same camera repeated observations (separate segments)
  Scenario D: Source time conflict / anomalous speed (ANOMALOUS classification)
  Scenario E: Missing coordinates (distance null, UNKNOWN plausibility)
  Scenario F: Probable OCR filtering (excluded by default, included with flag)
  Scenario G: Duplicate observation protection
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from src.journey import (
    CameraObservationSegment,
    CrossCameraObservation,
    JSONJourneyRepository,
    ObservationCorrelator,
    ObservationLoader,
    OrderingMode,
    PlausibilityAnalyzer,
    PlausibilityState,
    SequenceBuilder,
    VehicleJourney,
    VehicleJourneyBuilder,
    calculate_time_delta_seconds,
    can_order_globally,
    haversine_distance_meters,
)


# ════════════════════════════════════════════════════════════════════════════
# Scenario A: Chronologically Resolved Source Time
# ════════════════════════════════════════════════════════════════════════════

def test_scenario_a_chronological_source_time():
    """Scenario A: cam01 -> 10:00, cam07 -> 10:10, cam14 -> 10:20 produces ordered journey."""
    obs1 = CrossCameraObservation(
        observation_id="OBS-1", vehicle_id="VEH-001", registration_number="GJ01AB1234",
        normalized_registration_number="GJ01AB1234", camera_id="cam01", track_id="TRK-1",
        source_time="2026-08-31T10:00:00Z", source_time_status="RESOLVED",
        camera_latitude=23.0225, camera_longitude=72.5714, consensus_score=0.95, ocr_confidence=0.94,
    )
    obs2 = CrossCameraObservation(
        observation_id="OBS-2", vehicle_id="VEH-001", registration_number="GJ01AB1234",
        normalized_registration_number="GJ01AB1234", camera_id="cam07", track_id="TRK-2",
        source_time="2026-08-31T10:10:00Z", source_time_status="RESOLVED",
        camera_latitude=23.0500, camera_longitude=72.5800, consensus_score=0.92, ocr_confidence=0.91,
    )
    obs3 = CrossCameraObservation(
        observation_id="OBS-3", vehicle_id="VEH-001", registration_number="GJ01AB1234",
        normalized_registration_number="GJ01AB1234", camera_id="cam14", track_id="TRK-3",
        source_time="2026-08-31T10:20:00Z", source_time_status="RESOLVED",
        camera_latitude=23.0800, camera_longitude=72.5900, consensus_score=0.90, ocr_confidence=0.89,
    )

    builder = VehicleJourneyBuilder()
    journey = builder.build_from_observations([obs2, obs1, obs3])  # Unsorted input

    assert journey.ordering_mode == OrderingMode.SOURCE_TIME.value
    assert journey.status == "RECONSTRUCTED"
    assert len(journey.segments) == 3
    # Sorted order should be obs1 (10:00), obs2 (10:10), obs3 (10:20)
    assert [s.camera_id for s in journey.segments] == ["cam01", "cam07", "cam14"]

    assert len(journey.legs) == 2
    # Leg 1: cam01 -> cam07 (10 mins = 600s)
    leg1 = journey.legs[0]
    assert leg1.time_delta_seconds == 600.0
    assert leg1.straight_line_distance_m is not None
    assert leg1.implied_speed_kmh is not None
    assert leg1.plausibility in (PlausibilityState.PLAUSIBLE, PlausibilityState.POSSIBLE)

    # Leg 2: cam07 -> cam14 (10 mins = 600s)
    leg2 = journey.legs[1]
    assert leg2.time_delta_seconds == 600.0


# ════════════════════════════════════════════════════════════════════════════
# Scenario B: Camera-Local Media PTS Only (No Global Clock)
# ════════════════════════════════════════════════════════════════════════════

def test_scenario_b_camera_local_pts_only():
    """Scenario B: Unresolved source time produces camera-local sequence without cross-camera clock."""
    obs1 = CrossCameraObservation(
        observation_id="OBS-1", vehicle_id="VEH-001", registration_number="GJ01AB1234",
        normalized_registration_number="GJ01AB1234", camera_id="cam01", track_id="TRK-0001",
        first_seen_pts_ms=10240.0, recognition_pts_ms=11200.0, last_seen_pts_ms=12480.0,
        source_time=None, source_time_status="NOT_RESOLVED", consensus_score=0.94, ocr_confidence=0.93,
    )
    obs2 = CrossCameraObservation(
        observation_id="OBS-2", vehicle_id="VEH-001", registration_number="GJ01AB1234",
        normalized_registration_number="GJ01AB1234", camera_id="cam07", track_id="TRK-0019",
        first_seen_pts_ms=45200.0, recognition_pts_ms=46000.0, last_seen_pts_ms=47800.0,
        source_time=None, source_time_status="NOT_RESOLVED", consensus_score=0.92, ocr_confidence=0.91,
    )

    builder = VehicleJourneyBuilder()
    journey = builder.build_from_observations([obs1, obs2])

    assert journey.ordering_mode == OrderingMode.CAMERA_LOCAL.value
    assert journey.status == "OBSERVATION_SEQUENCE_ONLY"
    assert len(journey.segments) == 2

    # Cross-camera time delta must NOT be calculated from local PTS
    for leg in journey.legs:
        assert leg.time_delta_seconds is None
        assert leg.implied_speed_kmh is None
        assert leg.plausibility == PlausibilityState.UNKNOWN


# ════════════════════════════════════════════════════════════════════════════
# Scenario C: Same-Camera Repeated Observations
# ════════════════════════════════════════════════════════════════════════════

def test_scenario_c_same_camera_repeated_tracks():
    """Scenario C: Same camera repeated observations in different tracks are kept as distinct segments."""
    obs_trk1 = CrossCameraObservation(
        observation_id="OBS-1", vehicle_id="VEH-001", registration_number="GJ01AB1234",
        normalized_registration_number="GJ01AB1234", camera_id="cam01", track_id="TRK-0001",
        first_seen_pts_ms=10000.0, recognition_pts_ms=10500.0, last_seen_pts_ms=11000.0,
    )
    obs_trk2 = CrossCameraObservation(
        observation_id="OBS-2", vehicle_id="VEH-001", registration_number="GJ01AB1234",
        normalized_registration_number="GJ01AB1234", camera_id="cam01", track_id="TRK-0050",
        first_seen_pts_ms=80000.0, recognition_pts_ms=80500.0, last_seen_pts_ms=81000.0,
    )

    builder = VehicleJourneyBuilder()
    journey = builder.build_from_observations([obs_trk1, obs_trk2])

    assert len(journey.segments) == 2
    assert journey.segments[0].track_id == "TRK-0001"
    assert journey.segments[1].track_id == "TRK-0050"


# ════════════════════════════════════════════════════════════════════════════
# Scenario D: Source Time Conflict / Anomalous Speed
# ════════════════════════════════════════════════════════════════════════════

def test_scenario_d_anomalous_implied_speed():
    """Scenario D: Impossible speed (e.g. 500 km/h) or backwards time classified as ANOMALOUS."""
    obs1 = CrossCameraObservation(
        observation_id="OBS-1", vehicle_id="VEH-001", registration_number="GJ01AB1234",
        normalized_registration_number="GJ01AB1234", camera_id="cam01", track_id="TRK-1",
        source_time="2026-08-31T10:00:00Z", source_time_status="RESOLVED",
        camera_latitude=23.0000, camera_longitude=72.5000,
    )
    # 50 km distance in 10 seconds = 18,000 km/h!
    obs2 = CrossCameraObservation(
        observation_id="OBS-2", vehicle_id="VEH-001", registration_number="GJ01AB1234",
        normalized_registration_number="GJ01AB1234", camera_id="cam07", track_id="TRK-2",
        source_time="2026-08-31T10:00:10Z", source_time_status="RESOLVED",
        camera_latitude=23.4500, camera_longitude=72.5000,
    )

    builder = VehicleJourneyBuilder()
    journey = builder.build_from_observations([obs1, obs2])

    assert len(journey.legs) == 1
    leg = journey.legs[0]
    assert leg.plausibility == PlausibilityState.ANOMALOUS
    assert "exceeds" in leg.plausibility_reason


# ════════════════════════════════════════════════════════════════════════════
# Scenario E: Missing Coordinates
# ════════════════════════════════════════════════════════════════════════════

def test_scenario_e_missing_coordinates():
    """Scenario E: Missing coordinates produce null distance and UNKNOWN plausibility."""
    obs1 = CrossCameraObservation(
        observation_id="OBS-1", vehicle_id="VEH-001", registration_number="GJ01AB1234",
        normalized_registration_number="GJ01AB1234", camera_id="cam01", track_id="TRK-1",
        source_time="2026-08-31T10:00:00Z", source_time_status="RESOLVED",
        camera_latitude=None, camera_longitude=None,
    )
    obs2 = CrossCameraObservation(
        observation_id="OBS-2", vehicle_id="VEH-001", registration_number="GJ01AB1234",
        normalized_registration_number="GJ01AB1234", camera_id="cam07", track_id="TRK-2",
        source_time="2026-08-31T10:15:00Z", source_time_status="RESOLVED",
        camera_latitude=None, camera_longitude=None,
    )

    builder = VehicleJourneyBuilder()
    journey = builder.build_from_observations([obs1, obs2])

    assert len(journey.legs) == 1
    leg = journey.legs[0]
    assert leg.straight_line_distance_m is None
    assert leg.implied_speed_kmh is None
    assert leg.plausibility == PlausibilityState.UNKNOWN


# ════════════════════════════════════════════════════════════════════════════
# Scenario F: Probable OCR Filtering
# ════════════════════════════════════════════════════════════════════════════

def test_scenario_f_probable_ocr_filtering(tmp_path):
    """Scenario F: PROBABLE observations excluded by default, included when include_probable=True."""
    obs_file = tmp_path / "observations.jsonl"
    cam_file = tmp_path / "cameras.json"

    cam_file.write_text("[]", encoding="utf-8")

    rec_confirmed = {
        "observation_id": "OBS-1", "camera_id": "cam01", "track_id": "TRK-1",
        "registration_number": "GJ01AB1234", "normalized_registration_number": "GJ01AB1234",
        "status": "CONFIRMED", "consensus_score": 0.94, "ocr_confidence": 0.92,
    }
    rec_probable = {
        "observation_id": "OBS-2", "camera_id": "cam07", "track_id": "TRK-2",
        "registration_number": "GJ01AB1234", "normalized_registration_number": "GJ01AB1234",
        "status": "PROBABLE", "consensus_score": 0.81, "ocr_confidence": 0.82,
    }

    with open(obs_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec_confirmed) + "\n")
        f.write(json.dumps(rec_probable) + "\n")

    loader = ObservationLoader(observations_file=str(obs_file), cameras_file=str(cam_file))

    # Default: include_probable = False
    obs_default = loader.load_observations_for_vehicle("GJ01AB1234", include_probable=False)
    assert len(obs_default) == 1
    assert obs_default[0].recognition_status == "CONFIRMED"

    # Flag enabled: include_probable = True
    obs_with_prob = loader.load_observations_for_vehicle("GJ01AB1234", include_probable=True)
    assert len(obs_with_prob) == 2


# ════════════════════════════════════════════════════════════════════════════
# Scenario G: Duplicate Import & Intra-Track Aggregation
# ════════════════════════════════════════════════════════════════════════════

def test_scenario_g_intra_track_aggregation():
    """Multiple frame detections for the same camera track are aggregated into one segment."""
    frame1 = CrossCameraObservation(
        observation_id="OBS-1A", vehicle_id="VEH-001", registration_number="GJ01AB1234",
        normalized_registration_number="GJ01AB1234", camera_id="cam01", track_id="TRK-0001",
        first_seen_pts_ms=10240.0, recognition_pts_ms=10500.0, last_seen_pts_ms=11000.0,
        consensus_score=0.88, ocr_confidence=0.85, evidence_image="plate1.jpg",
    )
    frame2 = CrossCameraObservation(
        observation_id="OBS-1B", vehicle_id="VEH-001", registration_number="GJ01AB1234",
        normalized_registration_number="GJ01AB1234", camera_id="cam01", track_id="TRK-0001",
        first_seen_pts_ms=10240.0, recognition_pts_ms=11200.0, last_seen_pts_ms=12480.0,
        consensus_score=0.94, ocr_confidence=0.93, evidence_image="plate2_best.jpg",
    )

    correlator = ObservationCorrelator()
    segments = correlator.build_camera_segments([frame1, frame2])

    assert len(segments) == 1
    seg = segments[0]
    assert seg.camera_id == "cam01"
    assert seg.track_id == "TRK-0001"
    assert seg.first_seen_pts_ms == 10240.0
    assert seg.last_seen_pts_ms == 12480.0
    assert seg.recognition_pts_ms == 11200.0
    assert seg.consensus_score == 0.94
    assert seg.supporting_observation_count == 2
    assert seg.evidence_image == "plate2_best.jpg"


# ════════════════════════════════════════════════════════════════════════════
# Haversine Distance & Repository Persistence Tests
# ════════════════════════════════════════════════════════════════════════════

def test_haversine_distance_accuracy():
    # Distance between Ahmedabad Paldi Circle (~23.013, 72.562) and Chimanbhai Bridge (~23.056, 72.585)
    d = haversine_distance_meters(23.013, 72.562, 23.056, 72.585)
    assert d is not None
    assert 5000 < d < 6000  # ~5.3 km straight-line


def test_json_journey_repository(tmp_path):
    repo = JSONJourneyRepository(base_dir=str(tmp_path / "journeys"))
    seg = CameraObservationSegment(
        segment_id="SEG-1", vehicle_id="VEH-1", registration_number="GJ01AB1234",
        normalized_registration_number="GJ01AB1234", camera_id="cam01", track_id="TRK-1",
    )
    journey = VehicleJourney(
        journey_id="JRN-GJ01AB1234", vehicle_id="VEH-1", registration_number="GJ01AB1234",
        normalized_registration_number="GJ01AB1234", ordering_mode="CAMERA_LOCAL",
        status="OBSERVATION_SEQUENCE_ONLY", segments=[seg],
    )

    repo.save_journey(journey)
    loaded = repo.get_journey("GJ01AB1234")

    assert loaded is not None
    assert loaded.journey_id == "JRN-GJ01AB1234"
    assert len(loaded.segments) == 1
    assert (tmp_path / "journeys" / "reports" / "GJ01AB1234_journey.json").exists()
    assert (tmp_path / "journeys" / "reports" / "GJ01AB1234_journey.txt").exists()
