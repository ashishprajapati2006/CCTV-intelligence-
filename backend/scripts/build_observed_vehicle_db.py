#!/usr/bin/env python3
"""Step 7 - Build Observed Vehicle Database CLI.

Imports Step 6 ANPR consensus results from one or more camera streams into
the persistent Observed Vehicle Database.

Usage:
  # Ingest single camera results:
  python scripts/build_observed_vehicle_db.py --input data/anpr/cam01/consensus/anpr_consensus.json

  # Ingest multiple cameras:
  python scripts/build_observed_vehicle_db.py \
      --input data/anpr/cam01/consensus/anpr_consensus.json \
      --input data/anpr/cam02/consensus/anpr_consensus.json

  # Rebuild clean database:
  python scripts/build_observed_vehicle_db.py --rebuild --input data/anpr/cam01/consensus/anpr_consensus.json
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.logging import get_logger
from src.observed import (
    ANPRStep6Importer,
    JSONObservedVehicleRepository,
    VehicleAggregator,
)

logger = get_logger("build_observed_db")


def build_observed_database(
    input_files: List[str],
    storage_dir: str = "data/observed/vehicles",
    rebuild: bool = False,
) -> None:
    print("=" * 70)
    print("  GUJARAT POLICE CCTV - OBSERVED VEHICLE DATABASE INGESTION (STEP 7)")
    print("=" * 70)
    print(f"  Storage Directory : {storage_dir}")
    print(f"  Rebuild Mode      : {'YES (Wipe clean)' if rebuild else 'INCREMENTAL'}")
    print(f"  Input Files       : {len(input_files)}")
    for f in input_files:
        print(f"    - {f}")
    print("=" * 70)
    print("  NOTE: Observed Vehicle DB stores ground-truth CCTV observations.")
    print("        No watchlist matching or alert generation is performed here.")
    print("=" * 70 + "\n")

    p_storage = Path(storage_dir)
    if rebuild and p_storage.exists():
        logger.info("Rebuild requested. Clearing storage at %s...", storage_dir)
        for child in p_storage.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)

    repo = JSONObservedVehicleRepository(storage_dir=storage_dir)
    importer = ANPRStep6Importer(repository=repo)

    total_tracks_processed = 0
    total_obs_imported = 0
    total_duplicates_skipped = 0
    total_uncertain = 0

    for file_path in input_files:
        if not Path(file_path).exists():
            logger.warning("Input file not found, skipping: %s", file_path)
            continue

        res = importer.import_consensus_file(file_path)
        total_tracks_processed += res["tracks_processed"]
        total_obs_imported += res["observations_imported"]
        total_duplicates_skipped += res["observations_skipped_duplicate"]
        total_uncertain += res["uncertain_observations_recorded"]

    # Retrieve updated network statistics
    stats = repo.get_statistics()

    print("\n" + "=" * 70)
    print("  OBSERVED VEHICLE DATABASE -- INGESTION SUMMARY")
    print("=" * 70)
    print(f"  Total Step 6 Tracks Processed   : {total_tracks_processed}")
    print(f"  Observations Successfully Added : {total_obs_imported}")
    print(f"  Duplicate Observations Skipped  : {total_duplicates_skipped} (Idempotent)")
    print(f"  Uncertain / Unreadable Recorded : {total_uncertain}")
    print("-" * 70)
    print("  CURRENT DATABASE TOTALS:")
    print(f"  Total Unique Observed Vehicles  : {stats['total_observed_vehicles']}")
    print(f"  Total Vehicle Observations      : {stats['total_observations']}")
    print(f"  Total Preserved Tracks          : {stats['total_tracks']}")
    print(f"  Average Consensus Score         : {stats['average_consensus_score']:.4f}")
    print("-" * 70)
    print("  OBSERVATIONS BY CAMERA:")
    for cam_name, count in sorted(stats["observations_by_camera"].items()):
        print(f"    - {cam_name:<16}: {count}")
    print("-" * 70)
    print("  OBSERVATIONS BY RECOGNITION STATUS:")
    for stat_name, count in sorted(stats["observations_by_status"].items()):
        print(f"    - {stat_name:<16}: {count}")
    if stats["top_observed_vehicles"]:
        print("-" * 70)
        print("  TOP OBSERVED VEHICLES:")
        for v in stats["top_observed_vehicles"]:
            cams_str = ", ".join(v["cameras"])
            print(f"    - {v['registration_number']:<14}: {v['observation_count']} obs across {v['camera_count']} camera(s) [{cams_str}] (score: {v['best_consensus_score']:.2f})")
    print("=" * 70 + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Step 7 - Build Observed Vehicle Database from Step 6 ANPR Consensus"
    )
    ap.add_argument("--input", action="append", default=None,
                    help="Path to Step 6 anpr_consensus.json (repeatable for multiple cameras)")
    ap.add_argument("--scan-dir", default=None,
                    help="Scan directory recursively for anpr_consensus.json files (e.g. data/anpr)")
    ap.add_argument("--storage-dir", default="data/observed/vehicles",
                    help="Storage directory (default: data/observed/vehicles)")
    ap.add_argument("--rebuild", action="store_true",
                    help="Wipe and rebuild database from specified inputs")
    args = ap.parse_args()

    input_files = list(args.input) if args.input else []
    if args.scan_dir:
        p_scan = Path(args.scan_dir)
        if p_scan.exists():
            for p in sorted(p_scan.glob("**/consensus/anpr_consensus.json")):
                p_str = str(p)
                if p_str not in input_files:
                    input_files.append(p_str)

    if not input_files:
        ap.error("Must provide at least one --input file or a valid --scan-dir containing anpr_consensus.json files.")

    build_observed_database(
        input_files=input_files,
        storage_dir=args.storage_dir,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
