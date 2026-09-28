#!/usr/bin/env python3
"""JSON/JSONL to PostgreSQL + PostGIS Migration Script (Step 12).

Migrates historical file-backed JSON/JSONL datasets into PostgreSQL:
- Cameras catalogue (data/catalogue/normalized/cameras.json)
- Observed vehicles (data/observed/vehicles/vehicles.json)
- Vehicle observations (data/observed/vehicles/observations.jsonl)
- Vehicle tracks (data/observed/vehicles/tracks.json)
- Watchlist hotlist entries (data/watchlist/vehicles/watchlist.json)
- Matches (data/matches/raw/matches.jsonl)
- Reconstructed journeys (data/journeys/vehicles/*.json)

Requirements:
- Strictly idempotent (running twice does not create duplicate records)
- Preserves IDs, evidence paths, PTS, and source-time resolution status
- Preserves SYNTHETIC_DEMO flags
- Outputs detailed migration statistics

Usage:
    python scripts/migrate_json_to_postgres.py [--db-url DB_URL]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.orm import Session

from src.database.connection import (
    Base,
    get_database_url,
    get_db,
    init_engine,
    mask_db_url,
)
from src.database.repositories import (
    CameraRepository,
    JourneyRepository,
    MatchRepository,
    ObservationRepository,
    VehicleRepository,
    WatchlistRepository,
)


class JSONToPostgresMigrator:
    """ETL migrator for transferring file data to PostgreSQL."""

    def __init__(self, session: Session):
        self.session = session
        self.camera_repo = CameraRepository(session)
        self.vehicle_repo = VehicleRepository(session)
        self.obs_repo = ObservationRepository(session)
        self.watchlist_repo = WatchlistRepository(session)
        self.match_repo = MatchRepository(session)
        self.journey_repo = JourneyRepository(session)

        self.stats = {
            "cameras_migrated": 0,
            "vehicles_migrated": 0,
            "observations_migrated": 0,
            "anpr_observations_migrated": 0,
            "watchlist_entries_migrated": 0,
            "matches_migrated": 0,
            "journeys_migrated": 0,
            "journey_legs_migrated": 0,
            "records_rejected": 0,
            "duplicates_skipped": 0,
        }

    def migrate_cameras(self, path: Path) -> None:
        """Migrate cameras catalogue."""
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            cameras = json.load(f)

        for cam in cameras:
            if not cam.get("camera_id"):
                self.stats["records_rejected"] += 1
                continue
            try:
                self.camera_repo.upsert_camera(cam)
                self.stats["cameras_migrated"] += 1
            except Exception as e:
                self.stats["records_rejected"] += 1

    def migrate_vehicles(self, path: Path) -> None:
        """Migrate observed vehicles."""
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            vehicles = json.load(f)

        for v in vehicles:
            norm_reg = v.get("normalized_registration_number") or v.get("registration_number")
            if not norm_reg:
                self.stats["records_rejected"] += 1
                continue
            v["normalized_registration_number"] = norm_reg
            try:
                self.vehicle_repo.upsert_vehicle(v)
                self.stats["vehicles_migrated"] += 1
            except Exception:
                self.stats["records_rejected"] += 1

    def migrate_tracks(self, path: Path) -> None:
        """Migrate vehicle tracks."""
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            tracks = json.load(f)

        for t in tracks:
            if not t.get("track_id") or not t.get("camera_id"):
                self.stats["records_rejected"] += 1
                continue
            try:
                self.obs_repo.save_track(t)
            except Exception:
                self.stats["records_rejected"] += 1

    def migrate_observations(self, path: Path) -> None:
        """Migrate vehicle observations."""
        if not path.exists():
            return

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obs = json.loads(line)
                    obs_id = obs.get("observation_id")
                    if not obs_id:
                        self.stats["records_rejected"] += 1
                        continue

                    # Check if already present to accurately count duplicates
                    existing = self.obs_repo.get_observation(obs_id)
                    if existing:
                        self.stats["duplicates_skipped"] += 1
                        continue

                    self.obs_repo.add_observation(obs)
                    self.stats["observations_migrated"] += 1

                    # If ANPR fields present, also preserve raw ANPR observation
                    if obs.get("registration_number"):
                        self.obs_repo.add_anpr_observation({
                            "observation_id": obs_id,
                            "camera_id": obs.get("camera_id"),
                            "track_id": obs.get("track_id"),
                            "pts_ms": obs.get("recognition_pts_ms"),
                            "raw_text": obs.get("registration_number"),
                            "normalized_text": obs.get("normalized_registration_number"),
                            "ocr_confidence": obs.get("ocr_confidence"),
                            "plate_detection_confidence": obs.get("plate_detection_confidence"),
                            "plate_quality_score": obs.get("plate_quality_score"),
                            "image_crop_path": obs.get("evidence_image_path") or obs.get("evidence_image"),
                        })
                        self.stats["anpr_observations_migrated"] += 1

                except Exception:
                    self.stats["records_rejected"] += 1

    def migrate_watchlist(self, path: Path) -> None:
        """Migrate watchlist entries."""
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        for entry in entries:
            wl_id = entry.get("watchlist_id")
            if not wl_id:
                self.stats["records_rejected"] += 1
                continue
            try:
                self.watchlist_repo.upsert_entry(entry)
                self.stats["watchlist_entries_migrated"] += 1
            except Exception:
                self.stats["records_rejected"] += 1

    def migrate_matches(self, path: Path) -> None:
        """Migrate watchlist matches."""
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    m = json.loads(line)
                    match_id = m.get("match_id")
                    if not match_id:
                        self.stats["records_rejected"] += 1
                        continue

                    existing = self.match_repo.get_match(match_id)
                    if existing:
                        self.stats["duplicates_skipped"] += 1
                        continue

                    # Populate dummy watchlist_id if not present in un-matched record
                    if not m.get("watchlist_id"):
                        m["watchlist_id"] = "WL-NONE"

                    self.match_repo.add_match(m)
                    self.stats["matches_migrated"] += 1
                except Exception:
                    self.stats["records_rejected"] += 1

    def migrate_journeys(self, journeys_dir: Path) -> None:
        """Migrate vehicle journeys from data/journeys/vehicles/."""
        if not journeys_dir.exists():
            return

        for p in journeys_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    jdata = json.load(f)

                jid = jdata.get("journey_id")
                if not jid:
                    self.stats["records_rejected"] += 1
                    continue

                self.journey_repo.save_journey_data(jdata)
                self.stats["journeys_migrated"] += 1
                self.stats["journey_legs_migrated"] += len(jdata.get("legs", []))
            except Exception:
                self.stats["records_rejected"] += 1

    def reset_stats(self) -> None:
        """Reset counters for a new migration run."""
        self.stats = {
            "cameras_migrated": 0,
            "vehicles_migrated": 0,
            "observations_migrated": 0,
            "anpr_observations_migrated": 0,
            "watchlist_entries_migrated": 0,
            "matches_migrated": 0,
            "journeys_migrated": 0,
            "journey_legs_migrated": 0,
            "records_rejected": 0,
            "duplicates_skipped": 0,
        }

    def run_all(self) -> Dict[str, int]:
        """Execute full idempotent migration."""
        self.reset_stats()
        self.migrate_cameras(Path("data/catalogue/normalized/cameras.json"))
        self.migrate_vehicles(Path("data/observed/vehicles/vehicles.json"))
        self.migrate_tracks(Path("data/observed/vehicles/tracks.json"))
        self.migrate_observations(Path("data/observed/vehicles/observations.jsonl"))
        self.migrate_watchlist(Path("data/watchlist/vehicles/watchlist.json"))
        self.migrate_matches(Path("data/matches/raw/matches.jsonl"))
        self.migrate_journeys(Path("data/journeys/vehicles"))
        self.session.commit()
        return dict(self.stats)


def main():
    parser = argparse.ArgumentParser(description="Migrate JSON/JSONL datasets to PostgreSQL (Step 12)")
    parser.add_argument("--db-url", default=None, help="Database connection URL")
    args = parser.parse_args()

    url = args.db_url or get_database_url()
    masked = mask_db_url(url)

    print()
    print("=" * 70)
    print("  STEP 12 — JSON/JSONL TO POSTGRESQL DATA MIGRATION")
    print("=" * 70)
    print(f"  Target Database : {masked}")
    print()

    engine = init_engine(db_url=url)
    Base.metadata.create_all(engine)

    t0 = time.time()
    with Session(engine) as session:
        migrator = JSONToPostgresMigrator(session)
        stats = migrator.run_all()
    duration_ms = round((time.time() - t0) * 1000, 2)

    print("  ---------------- MIGRATION REPORT ----------------")
    print(f"  Cameras migrated             : {stats['cameras_migrated']}")
    print(f"  Vehicles migrated            : {stats['vehicles_migrated']}")
    print(f"  Observations migrated        : {stats['observations_migrated']}")
    print(f"  ANPR observations migrated   : {stats['anpr_observations_migrated']}")
    print(f"  Watchlist entries migrated   : {stats['watchlist_entries_migrated']}")
    print(f"  Matches migrated             : {stats['matches_migrated']}")
    print(f"  Journeys migrated            : {stats['journeys_migrated']}")
    print(f"  Journey legs migrated        : {stats['journey_legs_migrated']}")
    print(f"  Records rejected             : {stats['records_rejected']}")
    print(f"  Duplicate records skipped    : {stats['duplicates_skipped']}")
    print(f"  Total Migration Time         : {duration_ms} ms")
    print("  --------------------------------------------------")
    print()
    print("=" * 70)
    print("  MIGRATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
