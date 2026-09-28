#!/usr/bin/env python3
"""Database Initialization & PostGIS Extension Verification Script (Step 12).

Verifies database connectivity, enables PostGIS extension, builds all required
relational and spatial tables, and reports environment diagnostics.

Usage:
    python scripts/init_database.py [--db-url DB_URL] [--drop-existing]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from src.database.connection import (
    Base,
    check_postgis_status,
    get_database_url,
    init_engine,
    mask_db_url,
)
import src.database.models  # Register all models


def init_database(db_url: str | None = None, drop_existing: bool = False) -> bool:
    """Initialize database, extensions, and tables."""
    url = db_url or get_database_url()
    masked = mask_db_url(url)

    print()
    print("=" * 70)
    print("  STEP 12 — POSTGRESQL + POSTGIS DATABASE INITIALIZATION")
    print("=" * 70)
    print(f"  Target Database: {masked}")
    print()

    try:
        engine = init_engine(db_url=url)
    except Exception as e:
        print(f"  [ERROR] Failed to connect to database: {e}")
        return False

    status = check_postgis_status(engine)
    print("  --- Environment Diagnostics ---")
    print(f"  Dialect              : {'SQLite' if status['is_sqlite'] else 'PostgreSQL'}")
    print(f"  Engine Version       : {status['postgres_version']}")

    if status["is_sqlite"]:
        print("  PostGIS Extension    : N/A (SQLite mode; using internal spatial fallbacks)")
    else:
        # PostgreSQL: Attempt to enable PostGIS extension
        print("  Enabling PostGIS extension...")
        try:
            with engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
                conn.commit()
            # Re-check
            status = check_postgis_status(engine)
            if status["postgis_installed"]:
                print(f"  PostGIS Extension    : ENABLED (Version: {status['postgis_version']})")
            else:
                print("  [WARN] PostGIS extension could not be verified.")
        except Exception as e:
            print(f"  [WARN] Failed to execute 'CREATE EXTENSION postgis': {e}")
            print("         Spatial queries will fall back to spherical Haversine calculations.")

    print()

    # Drop tables if requested
    if drop_existing:
        print("  Dropping existing tables...")
        Base.metadata.drop_all(engine)
        print("  [+] Existing tables dropped.")

    # Create all tables
    print("  Creating relational and spatial tables...")
    Base.metadata.create_all(engine)

    tables = sorted(list(Base.metadata.tables.keys()))
    print(f"  [+] {len(tables)} tables verified:")
    for t in tables:
        print(f"      - {t}")

    print()
    print("=" * 70)
    print("  DATABASE INITIALIZATION COMPLETE")
    print("=" * 70)
    print()
    return True


def main():
    parser = argparse.ArgumentParser(description="Initialize PostgreSQL/PostGIS database (Step 12)")
    parser.add_argument("--db-url", default=None, help="Database connection URL")
    parser.add_argument("--drop-existing", action="store_true", help="Drop existing tables before recreating")
    args = parser.parse_args()

    success = init_database(db_url=args.db_url, drop_existing=args.drop_existing)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
