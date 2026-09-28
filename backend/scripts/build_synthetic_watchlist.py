#!/usr/bin/env python3
"""Step 8 - Build Representative Synthetic Watchlist CLI.

Reads Step 7 observed vehicles and generates a representative synthetic watchlist
containing:
  1. MATCHING records: Selects actual observed vehicle registrations from Step 7.
  2. NON-MATCHING records: Generates realistic valid Indian registrations that do
     not exist in the observed dataset.

Usage:
  python scripts/build_synthetic_watchlist.py \
      --observed data/observed/vehicles/vehicles.json \
      --match-count 2 \
      --nonmatch-count 5 \
      --seed 42 \
      --output data/watchlist/vehicles/watchlist.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.logging import get_logger
from src.watchlist import (
    JSONWatchlistRepository,
    SyntheticWatchlistGenerator,
    WatchlistValidator,
)

logger = get_logger("build_watchlist")


def main():
    ap = argparse.ArgumentParser(
        description="Step 8 - Generate Synthetic Watchlist from Step 7 Observed Vehicles"
    )
    ap.add_argument("--observed", default="data/observed/vehicles/vehicles.json",
                    help="Path to Step 7 vehicles.json (default: data/observed/vehicles/vehicles.json)")
    ap.add_argument("--match-count", type=int, default=5,
                    help="Number of observed vehicles to include as MATCHING records (default: 5)")
    ap.add_argument("--nonmatch-count", type=int, default=5,
                    help="Number of non-matching synthetic vehicles to generate (default: 5)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for deterministic generation (default: 42)")
    ap.add_argument("--output", default="data/watchlist/vehicles/watchlist.json",
                    help="Output path for vehicle watchlist (default: data/watchlist/vehicles/watchlist.json)")
    args = ap.parse_args()

    print("=" * 70)
    print("  GUJARAT POLICE CCTV - SYNTHETIC WATCHLIST BUILDER (STEP 8)")
    print("=" * 70)
    print(f"  Observed Vehicles Input : {args.observed}")
    print(f"  Requested Matches       : {args.match_count}")
    print(f"  Requested Non-Matches   : {args.nonmatch_count}")
    print(f"  Random Seed             : {args.seed}")
    print(f"  Target Output           : {args.output}")
    print("=" * 70)
    print("  DISCLAIMER: All generated records are strictly SYNTHETIC_DEMO.")
    print("              No real government databases or criminal records are used.")
    print("=" * 70 + "\n")

    generator = SyntheticWatchlistGenerator(seed=args.seed)
    watchlist, report = generator.generate_watchlist(
        observed_file_path=args.observed,
        match_count=args.match_count,
        nonmatch_count=args.nonmatch_count,
    )

    # Save to disk using repository
    repo = JSONWatchlistRepository(vehicles_path=args.output)
    repo.save_all(watchlist)

    # Audit the newly generated dataset
    validator = WatchlistValidator()
    observed_records = generator.load_observed_registrations(args.observed)
    observed_set = {
        r["normalized_registration_number"]
        for r in observed_records
        if r.get("normalized_registration_number")
    }
    val_res = validator.validate_dataset(watchlist, observed_registrations=observed_set)

    print("=" * 70)
    print("  WATCHLIST GENERATION REPORT")
    print("=" * 70)
    print(f"  Observed vehicles available     : {report['observed_vehicles_available']}")
    print()
    print(f"  Synthetic matching records      : {report['synthetic_matching_records']}")
    print(f"  Synthetic non-matching records  : {report['synthetic_nonmatching_records']}")
    print(f"  Total watchlist records         : {report['total_watchlist_records']}")
    print()
    print(f"  Expected match coverage         : {report['expected_match_coverage']}")
    print(f"  Seed used                       : {report['seed_used']}")
    print("-" * 70)
    print(f"  Validation Status               : {'PASSED (Valid)' if val_res['is_valid'] else 'FAILED'}")
    print(f"  Active Records                  : {val_res['active_records']}")
    print(f"  Inactive Records                : {val_res['inactive_records']}")
    print(f"  Duplicate Registrations         : {val_res['duplicate_count']}")
    print(f"  Invalid Formats                 : {val_res['invalid_format_count']}")
    print("-" * 70)
    print("  GUARANTEED MATCHING VEHICLES:")
    for m in val_res["guaranteed_matches"]:
        rec = repo.get_vehicle(m)
        print(f"    - {m:<14} [{rec.category if rec else 'N/A'}] (Priority: {rec.priority if rec else 'N/A'})")
    print("-" * 70)
    print("  NON-MATCHING CONTROL VEHICLES:")
    for nm in val_res["non_matching_records"]:
        rec = repo.get_vehicle(nm)
        print(f"    - {nm:<14} [{rec.category if rec else 'N/A'}] (Priority: {rec.priority if rec else 'N/A'}, Status: {rec.status if rec else 'N/A'})")
    print("=" * 70 + "\n")
    logger.info("Saved %d synthetic watchlist records to %s", len(watchlist), args.output)


if __name__ == "__main__":
    main()
