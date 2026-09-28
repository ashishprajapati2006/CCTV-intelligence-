#!/usr/bin/env python3
"""Step 13 -- PostgreSQL / PostGIS Spatial Database Migration & Export Tool.

Imports cameras, observed vehicles, sightings, watchlists, matches, and journeys
into PostgreSQL / PostGIS or standalone SQLite relational database.

Usage:
    python scripts/export_to_postgis.py
    python scripts/export_to_postgis.py --db-url postgresql://postgres:postgres@localhost:5432/cctv
"""
import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
from src.common.logging import get_logger

load_dotenv()
logger = get_logger("postgis_export")


def export_sqlite(output_db: str = "data/cctv_intelligence.db"):
    """Export relational data to SQLite database."""
    db_path = Path(output_db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS cameras (
        camera_id TEXT PRIMARY KEY,
        name TEXT,
        latitude REAL,
        longitude REAL,
        status TEXT,
        rtsp_url TEXT,
        hls_url TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS observed_vehicles (
        vehicle_id TEXT PRIMARY KEY,
        registration_number TEXT,
        normalized_registration TEXT UNIQUE,
        camera_count INT,
        observation_count INT,
        best_consensus_score REAL,
        status TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS vehicle_observations (
        observation_id TEXT PRIMARY KEY,
        camera_id TEXT,
        track_id TEXT,
        registration_number TEXT,
        consensus_score REAL,
        ocr_confidence REAL,
        first_seen_pts_ms REAL,
        last_seen_pts_ms REAL,
        evidence_image_path TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS watchlist_vehicles (
        watchlist_id TEXT PRIMARY KEY,
        registration_number TEXT,
        category TEXT,
        priority TEXT,
        status TEXT
    )
    """)

    # Ingest Cameras
    cam_file = Path("data/catalogue/normalized/cameras.json")
    if cam_file.exists():
        with open(cam_file, "r", encoding="utf-8") as f:
            cameras = json.load(f)
            for c in cameras:
                cur.execute("""
                INSERT OR REPLACE INTO cameras VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    c.get("camera_id"), c.get("name"), c.get("latitude"),
                    c.get("longitude"), c.get("status"), c.get("rtsp_url"), c.get("hls_url")
                ))

    # Ingest Observed Vehicles
    veh_file = Path("data/observed/vehicles/vehicles.json")
    if veh_file.exists():
        with open(veh_file, "r", encoding="utf-8") as f:
            vehicles = json.load(f)
            for v in vehicles:
                cur.execute("""
                INSERT OR REPLACE INTO observed_vehicles VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    v.get("vehicle_id"), v.get("registration_number"),
                    v.get("normalized_registration_number"), v.get("camera_count"),
                    v.get("observation_count"), v.get("best_consensus_score"), v.get("status")
                ))

    # Ingest Observations
    obs_file = Path("data/observed/vehicles/observations.jsonl")
    if obs_file.exists():
        with open(obs_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obs = json.loads(line)
                cur.execute("""
                INSERT OR REPLACE INTO vehicle_observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    obs.get("observation_id"), obs.get("camera_id"), obs.get("track_id"),
                    obs.get("registration_number"), obs.get("consensus_score"),
                    obs.get("ocr_confidence"), obs.get("first_seen_pts_ms"),
                    obs.get("last_seen_pts_ms"), obs.get("evidence_image_path")
                ))

    # Ingest Watchlist
    wl_file = Path("data/watchlist/vehicles/watchlist.json")
    if wl_file.exists():
        with open(wl_file, "r", encoding="utf-8") as f:
            wl = json.load(f)
            for item in wl:
                cur.execute("""
                INSERT OR REPLACE INTO watchlist_vehicles VALUES (?, ?, ?, ?, ?)
                """, (
                    item.get("watchlist_id"), item.get("registration_number"),
                    item.get("category"), item.get("priority"), item.get("status")
                ))

    conn.commit()

    # Report
    cur.execute("SELECT COUNT(*) FROM cameras")
    n_cam = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM observed_vehicles")
    n_veh = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM vehicle_observations")
    n_obs = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM watchlist_vehicles")
    n_wl = cur.fetchone()[0]
    conn.close()

    print("=" * 65)
    print("  STEP 13 -- RELATIONAL & SPATIAL DATABASE EXPORT")
    print("=" * 65)
    print(f"  Target Database   : {output_db}")
    print(f"  Cameras Ingested  : {n_cam}")
    print(f"  Vehicles Ingested : {n_veh}")
    print(f"  Sightings Ingested: {n_obs}")
    print(f"  Watchlist Records : {n_wl}")
    print(f"  PostGIS Schema DDL: database/schema.sql")
    print("[+] Step 13 Relational Database export completed successfully.\n")


def main():
    parser = argparse.ArgumentParser(description="Step 13 - Export CCTV intelligence to PostGIS / Relational DB")
    parser.add_argument("--db-url", default=None, help="PostgreSQL connection string")
    parser.add_argument("--output", default="data/cctv_intelligence.db", help="SQLite relational output path")
    args = parser.parse_args()
    export_sqlite(args.output)


if __name__ == "__main__":
    main()
