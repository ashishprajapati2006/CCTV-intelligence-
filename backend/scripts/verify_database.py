#!/usr/bin/env python3
"""Database Health & PostGIS Audit Tool (Step 12).

Verifies:
- PostgreSQL / SQLite connectivity
- PostGIS extension version and status
- Required relational tables existence
- Index creation and status
- Row counts across all authoritative entities
- Spatial column configuration

Usage:
    python scripts/verify_database.py [--db-url DB_URL]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, inspect, select, text
from sqlalchemy.orm import Session

from src.database.connection import (
    Base,
    check_postgis_status,
    get_database_url,
    init_engine,
    mask_db_url,
)
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


def verify_database(db_url: str | None = None) -> bool:
    """Run comprehensive database diagnostics."""
    url = db_url or get_database_url()
    masked = mask_db_url(url)

    print()
    print("=" * 70)
    print("  DATABASE HEALTH AUDIT — Step 12")
    print("=" * 70)
    print(f"  Target Database: {masked}")
    print()

    try:
        engine = init_engine(db_url=url)
    except Exception as e:
        print(f"  [FAIL] Cannot connect to database: {e}")
        return False

    status = check_postgis_status(engine)
    print("  DATABASE HEALTH")
    print("  ===============")
    print(f"  PostgreSQL: {'OK' if status['connected'] else 'FAIL'} ({status['postgres_version']})")
    print(f"  PostGIS   : {'OK (Version: ' + str(status['postgis_version']) + ')' if status['postgis_installed'] else 'N/A'}")
    print()

    insp = inspect(engine)
    existing_tables = set(insp.get_table_names())

    expected_models = [
        ("cameras", Camera),
        ("observed_vehicles", ObservedVehicle),
        ("vehicle_observations", VehicleObservation),
        ("anpr_observations", ANPRObservation),
        ("vehicle_tracks", VehicleTrack),
        ("watchlist_entries", WatchlistEntry),
        ("watchlist_matches", WatchlistMatch),
        ("alerts", AlertModel),
        ("vehicle_journeys", VehicleJourneyModel),
        ("journey_observations", JourneyObservationModel),
        ("journey_legs", JourneyLegModel),
    ]

    print("  TABLE & ROW AUDIT")
    print("  -----------------")
    all_ok = True

    with Session(engine) as session:
        for table_name, model in expected_models:
            if table_name in existing_tables:
                count = session.execute(select(func.count()).select_from(model)).scalar() or 0
                indices = [idx["name"] for idx in insp.get_indexes(table_name)]
                print(f"  [OK]   {table_name:<24}: {count:>6} rows | {len(indices)} index(es)")
            else:
                print(f"  [MISS] {table_name:<24}: NOT FOUND")
                all_ok = False

    print()
    print("  SPATIAL COLUMN AUDIT")
    print("  --------------------")
    if "cameras" in existing_tables:
        cols = {c["name"]: c for c in insp.get_columns("cameras")}
        geom_col = cols.get("geom")
        if geom_col:
            print(f"  [OK]   cameras.geom exists (Type: {geom_col['type']})")
        else:
            print("  [WARN] cameras.geom column not found")
        lat_col = cols.get("latitude")
        lon_col = cols.get("longitude")
        if lat_col and lon_col:
            print("  [OK]   cameras.latitude and cameras.longitude verified")

    print()
    print("=" * 70)
    print(f"  AUDIT RESULT: {'ALL SYSTEMS HEALTHY' if all_ok else 'ATTENTION REQUIRED'}")
    print("=" * 70)
    print()
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Verify database health and schema integrity")
    parser.add_argument("--db-url", default=None, help="Database connection URL")
    args = parser.parse_args()

    success = verify_database(db_url=args.db_url)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
