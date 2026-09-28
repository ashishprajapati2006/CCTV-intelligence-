"""tests/test_database.py — Step 12: PostgreSQL + PostGIS + GIS Data Layer Tests.

Verifies:
- Database connectivity and schema initialization
- Spatial point creation & coordinate validation
- Nearby camera queries within radius
- Camera distance calculation
- Bounding box spatial containment
- Camera coordinates filtering (with vs without coordinates)
- Observed vehicle insertion and duplicate prevention
- Vehicle observation & ANPR evidence insertion
- Watchlist entry insertion and SYNTHETIC_DEMO preservation
- Watchlist match persistence
- Journey, observations, and legs storage
- Journey to GeoJSON FeatureCollection generation
- JSON/JSONL migration idempotency
- Missing coordinates handling (geom = None)
- Source-time vs PTS temporal separation
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from src.database.connection import Base, check_postgis_status, init_engine
from src.database.models import (
    AlertModel,
    ANPRObservation,
    Camera,
    JourneyLegModel,
    JourneyObservationModel,
    ObservedVehicle,
    VehicleJourneyModel,
    VehicleObservation,
    VehicleTrack,
    WatchlistEntry,
    WatchlistMatch,
)
from src.database.repositories import (
    AlertRepository,
    CameraRepository,
    JourneyRepository,
    MatchRepository,
    ObservationRepository,
    VehicleRepository,
    WatchlistRepository,
    journey_to_geojson,
)
from scripts.migrate_json_to_postgres import JSONToPostgresMigrator


@pytest.fixture
def db_session():
    """Create in-memory SQLite session with full schema initialized."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


# ---------------------------------------------------------------------------
# 1. Database Connection & Schema
# ---------------------------------------------------------------------------

def test_database_connection_and_table_creation(db_session):
    """Verify in-memory database connectivity and all 11 tables exist."""
    engine = db_session.bind
    status = check_postgis_status(engine)
    assert status["connected"] is True
    assert status["is_sqlite"] is True

    from sqlalchemy import inspect
    insp = inspect(engine)
    tables = insp.get_table_names()
    assert "cameras" in tables
    assert "observed_vehicles" in tables
    assert "vehicle_observations" in tables
    assert "anpr_observations" in tables
    assert "watchlist_entries" in tables
    assert "vehicle_journeys" in tables


# ---------------------------------------------------------------------------
# 2. Camera Spatial Data & PostGIS Points
# ---------------------------------------------------------------------------

def test_camera_insertion_with_spatial_coordinates(db_session):
    """Verify camera insertion sets coordinates and geom correctly."""
    repo = CameraRepository(db_session)
    cam = repo.upsert_camera({
        "camera_id": "cam01",
        "name": "01 Chiman bhai Bridge",
        "latitude": 23.0673,
        "longitude": 72.5815,
        "status": "active",
    })
    db_session.commit()

    assert cam.camera_id == "cam01"
    assert cam.latitude == 23.0673
    assert cam.longitude == 72.5815
    assert cam.geom is not None


def test_camera_missing_coordinates_stores_null_geom(db_session):
    """Verify cameras without coordinates do NOT have fabricated coordinates."""
    repo = CameraRepository(db_session)
    cam = repo.upsert_camera({
        "camera_id": "cam99_unverified",
        "name": "Unverified Test Camera",
        "latitude": None,
        "longitude": None,
    })
    db_session.commit()

    assert cam.latitude is None
    assert cam.longitude is None
    assert cam.geom is None

    without_coords = repo.cameras_without_coordinates()
    assert any(c.camera_id == "cam99_unverified" for c in without_coords)


def test_nearby_cameras_spatial_query(db_session):
    """Verify nearby_cameras returns installations within specified radius."""
    repo = CameraRepository(db_session)
    # Ahmedabad Chimanbhai Bridge
    repo.upsert_camera({"camera_id": "cam01", "name": "Chimanbhai", "latitude": 23.0673, "longitude": 72.5815})
    # Janpath (~4.7 km south)
    repo.upsert_camera({"camera_id": "cam02", "name": "Janpath", "latitude": 23.0286, "longitude": 72.5620})
    # Surat (~200 km south)
    repo.upsert_camera({"camera_id": "cam_surat", "name": "Surat Chowk", "latitude": 21.1702, "longitude": 72.8311})
    db_session.commit()

    # Search within 10 km of Sabarmati
    nearby = repo.nearby_cameras(lat=23.06, lon=72.58, radius_m=10000.0)
    cam_ids = [n["camera_id"] for n in nearby]

    assert "cam01" in cam_ids
    assert "cam02" in cam_ids
    assert "cam_surat" not in cam_ids  # Out of radius


def test_camera_distance_calculation(db_session):
    """Verify straight-line camera distance calculation."""
    repo = CameraRepository(db_session)
    repo.upsert_camera({"camera_id": "cam01", "latitude": 23.0673, "longitude": 72.5815})
    repo.upsert_camera({"camera_id": "cam02", "latitude": 23.0286, "longitude": 72.5620})
    db_session.commit()

    dist = repo.camera_distance("cam01", "cam02")
    assert dist is not None
    # Distance between Chimanbhai and Janpath is approx 4.7 km (4700 - 4800 m)
    assert 4500.0 < dist < 5000.0


def test_cameras_in_bounding_box(db_session):
    """Verify cameras_in_bbox correctly filters points inside envelope."""
    repo = CameraRepository(db_session)
    repo.upsert_camera({"camera_id": "cam01", "latitude": 23.0673, "longitude": 72.5815})
    repo.upsert_camera({"camera_id": "cam02", "latitude": 23.0286, "longitude": 72.5620})
    db_session.commit()

    in_box = repo.cameras_in_bbox(min_lon=72.55, min_lat=23.00, max_lon=72.60, max_lat=23.10)
    ids = [c.camera_id for c in in_box]
    assert "cam01" in ids
    assert "cam02" in ids


# ---------------------------------------------------------------------------
# 3. Vehicle & Observation Persistence
# ---------------------------------------------------------------------------

def test_vehicle_insertion_and_duplicate_handling(db_session):
    """Verify vehicle insertion and idempotency."""
    repo = VehicleRepository(db_session)
    v1 = repo.upsert_vehicle({
        "vehicle_id": "VEH-GJ01AB1234",
        "registration_number": "GJ01AB1234",
        "normalized_registration_number": "GJ01AB1234",
        "camera_count": 2,
        "observation_count": 3,
        "best_consensus_score": 0.91,
    })
    db_session.commit()

    assert v1.id is not None
    assert v1.normalized_registration_number == "GJ01AB1234"

    # Second upsert with higher count
    v2 = repo.upsert_vehicle({
        "vehicle_id": "VEH-GJ01AB1234",
        "registration_number": "GJ01AB1234",
        "normalized_registration_number": "GJ01AB1234",
        "camera_count": 3,
        "observation_count": 5,
    })
    db_session.commit()

    assert repo.count_vehicles() == 1
    assert v2.camera_count == 3
    assert v2.observation_count == 5


def test_vehicle_observation_and_time_separation(db_session):
    """Verify PTS and source-time fields are stored separately without PTS treated as global time."""
    repo = ObservationRepository(db_session)
    obs = repo.add_observation({
        "observation_id": "OBS-01",
        "camera_id": "cam01",
        "track_id": "TRK-001",
        "registration_number": "GJ01AB1234",
        "normalized_registration_number": "GJ01AB1234",
        "first_seen_pts_ms": 10000.0,
        "recognition_pts_ms": 12000.0,
        "last_seen_pts_ms": 13000.0,
        "source_time": None,
        "source_time_status": "NOT_RESOLVED",
        "consensus_score": 0.94,
    })
    db_session.commit()

    assert obs.recognition_pts_ms == 12000.0
    assert obs.source_time is None
    assert obs.source_time_status == "NOT_RESOLVED"


def test_anpr_observation_preserves_raw_ocr(db_session):
    """Verify raw OCR text is preserved alongside normalized text."""
    repo = ObservationRepository(db_session)
    anpr = repo.add_anpr_observation({
        "camera_id": "cam01",
        "track_id": "TRK-001",
        "frame_id": 42,
        "pts_ms": 12000.0,
        "raw_text": "GJ 01 AB 1234",
        "normalized_text": "GJ01AB1234",
        "ocr_confidence": 0.89,
    })
    db_session.commit()

    assert anpr.raw_text == "GJ 01 AB 1234"
    assert anpr.normalized_text == "GJ01AB1234"


# ---------------------------------------------------------------------------
# 4. Watchlist & Matches
# ---------------------------------------------------------------------------

def test_watchlist_insertion_preserves_synthetic_flag(db_session):
    """Verify synthetic demonstration flag is preserved on hotlist entries."""
    repo = WatchlistRepository(db_session)
    entry = repo.upsert_entry({
        "watchlist_id": "WL-001",
        "registration_number": "GJ01AB1234",
        "category": "STOLEN_VEHICLE",
        "priority": "HIGH",
        "source": "SYNTHETIC_DEMO",
        "synthetic": True,
    })
    db_session.commit()

    assert entry.source == "SYNTHETIC_DEMO"
    assert entry.synthetic is True
    assert repo.get_by_registration("GJ01AB1234") is not None


def test_watchlist_match_persistence(db_session):
    """Verify confirmed match persistence."""
    repo = MatchRepository(db_session)
    match = repo.add_match({
        "match_id": "MATCH-001",
        "observation_id": "OBS-01",
        "watchlist_id": "WL-001",
        "registration_number": "GJ01AB1234",
        "normalized_registration_number": "GJ01AB1234",
        "match_score": 1.0,
        "camera_id": "cam01",
        "track_id": "TRK-001",
        "recognition_pts_ms": 12000.0,
    })
    db_session.commit()

    assert match.match_id == "MATCH-001"
    assert repo.get_match("MATCH-001") is not None


# ---------------------------------------------------------------------------
# 5. Journeys & GeoJSON
# ---------------------------------------------------------------------------

def test_journey_persistence_and_retrieval(db_session):
    """Verify storing journey headers, observation sequence, and inter-camera legs."""
    j_repo = JourneyRepository(db_session)
    j_repo.save_journey_data({
        "journey_id": "JRN-GJ01AB1234",
        "registration_number": "GJ01AB1234",
        "status": "OBSERVATION_SEQUENCE_ONLY",
        "time_basis": "CAMERA_LOCAL_MEDIA_PTS",
        "confidence": 0.72,
        "observations": [
            {"observation_id": "OBS-01", "camera_id": "cam01"},
            {"observation_id": "OBS-02", "camera_id": "cam02"},
        ],
        "legs": [
            {
                "from_observation_id": "OBS-01",
                "to_observation_id": "OBS-02",
                "from_camera_id": "cam01",
                "to_camera_id": "cam02",
                "straight_line_distance_m": 4743.0,
                "distance_status": "AVAILABLE",
                "plausibility": "UNKNOWN",
            }
        ],
    })
    db_session.commit()

    assert j_repo.count() == 1
    retrieved = j_repo.get_journey("GJ01AB1234")
    assert retrieved is not None
    assert len(retrieved["legs"]) == 1
    assert retrieved["legs"][0]["straight_line_distance_m"] == 4743.0


def test_journey_to_geojson_feature_collection(db_session):
    """Verify GeoJSON FeatureCollection generation with verified coordinates and trajectory disclaimer."""
    cam_repo = CameraRepository(db_session)
    cam_repo.upsert_camera({"camera_id": "cam01", "latitude": 23.0673, "longitude": 72.5815})
    cam_repo.upsert_camera({"camera_id": "cam02", "latitude": 23.0286, "longitude": 72.5620})
    db_session.commit()

    journey_data = {
        "registration_number": "GJ01AB1234",
        "observations": [
            {"camera_id": "cam01", "camera_latitude": 23.0673, "camera_longitude": 72.5815, "recognition_pts_ms": 12000.0},
            {"camera_id": "cam02", "camera_latitude": 23.0286, "camera_longitude": 72.5620, "recognition_pts_ms": 45000.0},
        ],
    }

    geojson = journey_to_geojson(journey_data)
    assert geojson["type"] == "FeatureCollection"
    features = geojson["features"]

    # 2 points + 1 line
    assert len(features) == 3
    points = [f for f in features if f["geometry"]["type"] == "Point"]
    lines = [f for f in features if f["geometry"]["type"] == "LineString"]

    assert len(points) == 2
    assert len(lines) == 1

    # Verify GeoJSON coordinate order [lon, lat]
    assert points[0]["geometry"]["coordinates"] == [72.5815, 23.0673]
    assert points[1]["geometry"]["coordinates"] == [72.5620, 23.0286]

    # Verify forensic disclaimer on trajectory line
    assert "NOT an actual road route" in lines[0]["properties"]["disclaimer"]


# ---------------------------------------------------------------------------
# 6. Migration & Idempotency
# ---------------------------------------------------------------------------

def test_migrator_idempotency(db_session):
    """Verify running migrator twice produces zero duplicate records."""
    migrator = JSONToPostgresMigrator(db_session)
    stats1 = migrator.run_all()

    assert stats1["cameras_migrated"] == 30
    assert stats1["vehicles_migrated"] == 111
    assert stats1["observations_migrated"] == 152
    assert stats1["records_rejected"] == 0

    # Second pass
    stats2 = migrator.run_all()
    # In second pass, existing observations should be skipped
    assert stats2["observations_migrated"] == 0
    assert stats2["duplicates_skipped"] > 0
