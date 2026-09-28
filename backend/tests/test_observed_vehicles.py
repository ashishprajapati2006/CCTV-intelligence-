"""Unit tests for Step 7 — Observed Vehicle Database with strict timestamp semantics.

Covers:
  1. Step 6 first_seen_pts_ms is preserved.
  2. Step 6 last_seen_pts_ms is preserved.
  3. Evidence PTS is preserved.
  4. recognition_pts_ms comes from evidence, not datetime.now().
  5. ingested_at_utc is allowed to use current application time.
  6. source_time remains null when not resolvable.
  7. Step 7 never converts PTS to wall-clock using datetime.now()+PTS.
  8. Cross-camera views do not compare PTS directly (Common-Clock Rule).
  9. Observation IDs remain idempotent and deterministic.
  10. Re-importing the same Step 6 record does not create duplicates.
  11. Evidence filename PTS discrepancy auditing.
  12. Uncertain / unreadable routing without promotion.
  13. Security boundary (no watchlist / wanted fields).
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from src.observed import (
    ANPRStep6Importer,
    JSONObservedVehicleRepository,
    ObservedVehicle,
    ObservedVehicleQueries,
    SightingRecord,
    SupportingFrameRef,
    TimelineEntry,
    UncertainObservation,
    VehicleAggregator,
    VehicleObservation,
    VehicleTrackRecord,
)
from src.observed.importer import check_evidence_filename_pts


@pytest.fixture
def temp_repo(tmp_path):
    storage_dir = tmp_path / "observed_db"
    return JSONObservedVehicleRepository(storage_dir=str(storage_dir))


def _make_dummy_obs(
    cam_id: str = "cam01",
    track_id: str = "TRK-0001",
    reg: str = "GJ01AB1234",
    first_pts: float = 1000.0,
    rec_pts: float = 1200.0,
    last_pts: float = 1500.0,
    score: float = 0.90,
    status: str = "CONFIRMED",
) -> VehicleObservation:
    return VehicleObservation(
        observation_id=f"OBS-{cam_id}-{track_id}-{int(rec_pts)}",
        camera_id=cam_id,
        track_id=track_id,
        registration_number=reg,
        normalized_registration_number=reg,
        first_seen_pts_ms=first_pts,
        recognition_pts_ms=rec_pts,
        last_seen_pts_ms=last_pts,
        source_time=None,
        source_time_status="NOT_RESOLVED",
        ocr_confidence=0.92,
        plate_detection_confidence=0.95,
        plate_quality_score=0.88,
        consensus_score=score,
        supporting_frame_count=3,
        status=status,
        evidence_image="data/snapshots/cam01/plates/sample_pts1200.jpg",
        evidence_filename_pts_status="MATCH",
        evidence_filename_pts_ms=1200.0,
        source="sentinel",
    )


# ════════════════════════════════════════════════════════════════════════════
# 1. Observation Deduplication & Idempotency
# ════════════════════════════════════════════════════════════════════════════

def test_add_observation_deduplication(temp_repo):
    obs1 = _make_dummy_obs(rec_pts=1000.0)
    obs2 = _make_dummy_obs(rec_pts=1200.0)  # Same 1-second bucket (1s)

    added1 = temp_repo.add_observation(obs1)
    added2 = temp_repo.add_observation(obs2)

    assert added1 is True
    assert added2 is False  # Duplicate skipped
    assert temp_repo.count_observations() == 1


def test_distinct_observations_saved(temp_repo):
    obs1 = _make_dummy_obs(rec_pts=1000.0)
    obs2 = _make_dummy_obs(rec_pts=5000.0)  # Different bucket

    added1 = temp_repo.add_observation(obs1)
    added2 = temp_repo.add_observation(obs2)

    assert added1 is True
    assert added2 is True
    assert temp_repo.count_observations() == 2


# ════════════════════════════════════════════════════════════════════════════
# 2. Vehicle Aggregation & Common-Clock Rule
# ════════════════════════════════════════════════════════════════════════════

def test_aggregate_single_camera_multiple_sightings():
    agg = VehicleAggregator()
    obs_list = [
        _make_dummy_obs(cam_id="cam01", track_id="TRK-001", first_pts=1000.0, rec_pts=1200.0, last_pts=1500.0, score=0.85),
        _make_dummy_obs(cam_id="cam01", track_id="TRK-001", first_pts=2000.0, rec_pts=2200.0, last_pts=2500.0, score=0.92),
        _make_dummy_obs(cam_id="cam01", track_id="TRK-002", first_pts=5000.0, rec_pts=5200.0, last_pts=5500.0, score=0.88),
    ]

    veh = agg.aggregate_vehicle_observations("GJ01AB1234", obs_list)
    assert veh is not None
    assert veh.normalized_registration_number == "GJ01AB1234"
    assert veh.observation_count == 3
    assert veh.track_count == 2
    assert veh.camera_count == 1
    assert veh.cameras == ["cam01"]
    assert veh.first_seen.pts_ms == 1000.0
    assert veh.last_seen.pts_ms == 5500.0
    assert veh.first_seen.source_time is None
    assert veh.first_seen.source_time_status == "NOT_RESOLVED"
    assert veh.best_consensus_score == 0.92
    assert len(veh.timeline) == 3


def test_cross_camera_aggregation_common_clock_rule():
    """Verify PTS is not directly compared across independent camera clocks."""
    agg = VehicleAggregator()
    # cam07 PTS = 5.0s, cam01 PTS = 10.0s
    # Ingestion order: cam01 first, then cam07.
    # Without source_time, cam07 PTS=5 must NOT be treated as earlier than cam01 PTS=10!
    obs_list = [
        _make_dummy_obs(cam_id="cam01", track_id="TRK-001", first_pts=10000.0, rec_pts=11200.0, last_pts=12480.0, score=0.90),
        _make_dummy_obs(cam_id="cam07", track_id="TRK-019", first_pts=5000.0, rec_pts=6000.0, last_pts=7800.0, score=0.94),
    ]

    veh = agg.aggregate_vehicle_observations("GJ01AB1234", obs_list)
    assert veh is not None
    assert veh.camera_count == 2
    assert veh.cameras == ["cam01", "cam07"]
    # Grouped by camera order
    assert veh.timeline[0].camera_id == "cam01"
    assert veh.timeline[1].camera_id == "cam07"
    assert veh.timeline[0].first_seen_pts_ms == 10000.0
    assert veh.timeline[1].first_seen_pts_ms == 5000.0
    assert veh.first_seen.source_time_status == "NOT_RESOLVED"


# ════════════════════════════════════════════════════════════════════════════
# 3. Importing Step 6 Output & Timestamp Preservation
# ════════════════════════════════════════════════════════════════════════════

def test_importer_step6_preserves_pts_and_evidence(tmp_path, temp_repo):
    step6_data = {
        "camera_id": "cam01",
        "tracks": [
            {
                "camera_id": "cam01",
                "track_id": "TRK-0001",
                "final_registration_number": "GJ01AB1234",
                "recognition_status": "CONFIRMED",
                "consensus_score": 0.94,
                "first_seen_pts_ms": 10240.0,
                "last_seen_pts_ms": 12480.0,
                "supporting_frame_count": 3,
                "evidence_chain": [
                    {
                        "camera_id": "cam01",
                        "track_id": "TRK-0001",
                        "frame_id": 18,
                        "pts_ms": 10500.0,
                        "plate_detection_confidence": 0.80,
                        "plate_quality_score": 0.70,
                        "image_crop_path": "data/snapshots/cam01/plates/p1_pts10500.jpg",
                        "raw_ocr": {"raw_text": "GJ01AB1234", "confidence": 0.75},
                    },
                    {
                        "camera_id": "cam01",
                        "track_id": "TRK-0001",
                        "frame_id": 24,
                        "pts_ms": 11200.0,  # Strongest evidence frame
                        "plate_detection_confidence": 0.95,
                        "plate_quality_score": 0.88,
                        "image_crop_path": "data/snapshots/cam01/plates/p2_pts11200.jpg",
                        "raw_ocr": {"raw_text": "GJ01AB1234", "confidence": 0.93},
                    }
                ]
            }
        ]
    }

    in_file = tmp_path / "anpr_consensus.json"
    with open(in_file, "w", encoding="utf-8") as f:
        json.dump(step6_data, f)

    importer = ANPRStep6Importer(repository=temp_repo)
    res = importer.import_consensus_file(str(in_file))

    assert res["observations_imported"] == 1
    obs_list = temp_repo.get_vehicle_observations("GJ01AB1234")
    assert len(obs_list) == 1
    obs = obs_list[0]

    # Check PTS preservation
    assert obs.first_seen_pts_ms == 10240.0
    assert obs.last_seen_pts_ms == 12480.0
    assert obs.recognition_pts_ms == 11200.0  # From strongest evidence frame!
    assert obs.source_time is None
    assert obs.source_time_status == "NOT_RESOLVED"
    assert obs.evidence_filename_pts_status == "MATCH"
    assert obs.ingested_at_utc is not None  # application time only

    # Check track record preservation
    trk = temp_repo.get_track("TRK-0001")
    assert trk is not None
    assert trk.first_seen_pts_ms == 10240.0
    assert trk.recognition_pts_ms == 11200.0
    assert trk.last_seen_pts_ms == 12480.0


def test_importer_filename_pts_discrepancy(tmp_path, temp_repo):
    """Detect when evidence frame PTS does not match embedded filename PTS."""
    step6_data = {
        "camera_id": "cam01",
        "tracks": [
            {
                "camera_id": "cam01",
                "track_id": "TRK-0001",
                "final_registration_number": "GJ01AB1234",
                "recognition_status": "CONFIRMED",
                "consensus_score": 0.94,
                "first_seen_pts_ms": 10240.0,
                "last_seen_pts_ms": 12480.0,
                "evidence_chain": [
                    {
                        "camera_id": "cam01",
                        "track_id": "TRK-0001",
                        "frame_id": 24,
                        "pts_ms": 11200.0,  # 11200.0 ms
                        "image_crop_path": "data/snapshots/cam01/plates/cam01_TRK-0001_frame000024_pts1520_plate08.jpg",  # pts1520
                        "raw_ocr": {"raw_text": "GJ01AB1234", "confidence": 0.93},
                    }
                ]
            }
        ]
    }

    in_file = tmp_path / "anpr_consensus.json"
    with open(in_file, "w", encoding="utf-8") as f:
        json.dump(step6_data, f)

    importer = ANPRStep6Importer(repository=temp_repo)
    importer.import_consensus_file(str(in_file))

    obs = temp_repo.get_vehicle_observations("GJ01AB1234")[0]
    assert obs.recognition_pts_ms == 11200.0
    assert obs.evidence_filename_pts_ms == 1520.0
    assert obs.evidence_filename_pts_status == "DISCREPANCY"


def test_check_evidence_filename_pts_helper():
    assert check_evidence_filename_pts("frame24_pts11200.jpg", 11200.0) == ("MATCH", 11200.0)
    assert check_evidence_filename_pts("frame24_pts1520.jpg", 11200.0) == ("DISCREPANCY", 1520.0)
    assert check_evidence_filename_pts("frame24_no_pts.jpg", 11200.0) == ("UNKNOWN", None)
    assert check_evidence_filename_pts(None, 11200.0) == ("UNKNOWN", None)


def test_importer_idempotency(tmp_path, temp_repo):
    """Running import twice on the exact same input must NOT duplicate records."""
    step6_data = {
        "camera_id": "cam01",
        "tracks": [
            {
                "camera_id": "cam01",
                "track_id": "TRK-0001",
                "final_registration_number": "GJ01AB1234",
                "recognition_status": "CONFIRMED",
                "consensus_score": 0.91,
                "first_seen_pts_ms": 1120.0,
                "last_seen_pts_ms": 1920.0,
                "evidence_chain": []
            }
        ]
    }

    in_file = tmp_path / "anpr_consensus.json"
    with open(in_file, "w", encoding="utf-8") as f:
        json.dump(step6_data, f)

    importer = ANPRStep6Importer(repository=temp_repo)
    importer.import_consensus_file(str(in_file))

    obs_count_1 = temp_repo.count_observations()
    veh_count_1 = len(temp_repo.list_vehicles())

    # Second run
    res2 = importer.import_consensus_file(str(in_file))

    assert res2["observations_imported"] == 0
    assert res2["observations_skipped_duplicate"] == 1
    assert temp_repo.count_observations() == obs_count_1
    assert len(temp_repo.list_vehicles()) == veh_count_1


# ════════════════════════════════════════════════════════════════════════════
# 4. Uncertain Recognition Handling
# ════════════════════════════════════════════════════════════════════════════

def test_uncertain_recognition_not_promoted(temp_repo):
    importer = ANPRStep6Importer(repository=temp_repo)
    step6_data = {
        "camera_id": "cam01",
        "tracks": [
            {
                "camera_id": "cam01",
                "track_id": "TRK-0003",
                "final_registration_number": None,
                "recognition_status": "UNCERTAIN",
                "consensus_score": 0.42,
                "first_seen_pts_ms": 1000.0,
                "evidence_chain": [
                    {"raw_ocr": {"raw_text": "GJ01A8I234", "confidence": 0.35}}
                ]
            }
        ]
    }
    tmp_f = temp_repo.storage_dir / "test_uncertain.json"
    with open(tmp_f, "w", encoding="utf-8") as f:
        json.dump(step6_data, f)

    res = importer.import_consensus_file(str(tmp_f))
    assert res["observations_imported"] == 0
    assert res["uncertain_observations_recorded"] == 1
    assert len(temp_repo.list_vehicles()) == 0


# ════════════════════════════════════════════════════════════════════════════
# 5. Queries & Security Separation
# ════════════════════════════════════════════════════════════════════════════

def test_queries_service(temp_repo):
    obs = _make_dummy_obs(cam_id="cam01", reg="GJ05CD5678")
    temp_repo.add_observation(obs)
    agg = VehicleAggregator()
    veh = agg.aggregate_vehicle_observations("GJ05CD5678", [obs])
    temp_repo.save_vehicle(veh)

    queries = ObservedVehicleQueries(repository=temp_repo)

    fetched_veh = queries.get_vehicle("gj-05 cd 5678")
    assert fetched_veh is not None
    assert fetched_veh.normalized_registration_number == "GJ05CD5678"

    cams = queries.get_vehicle_cameras("GJ05CD5678")
    assert cams == ["cam01"]

    timeline = queries.get_vehicle_timeline("GJ05CD5678")
    assert len(timeline) == 1
    assert timeline[0]["first_seen_pts_ms"] == 1000.0
    assert timeline[0]["recognition_pts_ms"] == 1200.0

    stats = queries.get_statistics()
    assert stats["total_observed_vehicles"] == 1
    assert stats["total_observations"] == 1


def test_security_separation_no_watchlist_fields():
    """ObservedVehicle schema must NOT contain watchlist/wanted fields."""
    veh = ObservedVehicle(
        vehicle_id="VEH-000001",
        registration_number="GJ01AB1234",
        normalized_registration_number="GJ01AB1234",
        first_seen=SightingRecord(camera_id="cam01", pts_ms=1000.0),
        last_seen=SightingRecord(camera_id="cam01", pts_ms=1000.0),
        camera_count=1,
        cameras=["cam01"],
        observation_count=1,
        track_count=1,
        best_consensus_score=0.91,
        average_consensus_score=0.91,
        status="OBSERVED",
    )
    d = veh.to_dict()
    assert "wanted" not in d
    assert "suspicious" not in d
    assert "watchlist" not in d
    assert "alert" not in d
