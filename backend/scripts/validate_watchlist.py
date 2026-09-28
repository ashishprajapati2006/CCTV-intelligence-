#!/usr/bin/env python3
"""Step 8 - Validate Synthetic Watchlist CLI.

Audits a watchlist file against format rules, duplicate constraints, and
cross-references against the Step 7 Observed Vehicle Database to verify expected
demonstration coverage.

Usage:
  python scripts/validate_watchlist.py \
      --watchlist data/watchlist/vehicles/watchlist.json \
      --observed data/observed/vehicles/vehicles.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.watchlist import (
    JSONWatchlistRepository,
    WatchlistValidator,
)


def main():
    ap = argparse.ArgumentParser(description="Step 8 - Validate Watchlist Integrity and Match Coverage")
    ap.add_argument("--watchlist", default="data/watchlist/vehicles/watchlist.json",
                    help="Path to vehicle watchlist (default: data/watchlist/vehicles/watchlist.json)")
    ap.add_argument("--observed", default="data/observed/vehicles/vehicles.json",
                    help="Path to Step 7 vehicles.json (default: data/observed/vehicles/vehicles.json)")
    ap.add_argument("--json", action="store_true",
                    help="Output results as JSON")
    args = ap.parse_args()

    wl_file = Path(args.watchlist)
    if not wl_file.exists():
        print(f"[ERROR] Watchlist file not found: {args.watchlist}")
        sys.exit(1)

    repo = JSONWatchlistRepository(vehicles_path=str(wl_file))
    vehicles = repo.all_vehicles()

    # Load observed registrations
    observed_regs = set()
    obs_file = Path(args.observed)
    if obs_file.exists():
        try:
            with open(obs_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for r in data:
                        if r.get("normalized_registration_number"):
                            observed_regs.add(r["normalized_registration_number"])
        except Exception as e:
            print(f"[WARN] Failed to read observed vehicles from {args.observed}: {e}")

    validator = WatchlistValidator()
    report = validator.validate_dataset(vehicles, observed_registrations=observed_regs)

    if args.json:
        print(json.dumps(report, indent=2))
        sys.exit(0 if report["is_valid"] else 1)

    print("=" * 70)
    print("  WATCHLIST VALIDATION & MATCH COVERAGE AUDIT")
    print("=" * 70)
    print(f"  Target Watchlist File       : {args.watchlist}")
    print(f"  Observed Vehicles File      : {args.observed}")
    print(f"  Overall Validation Status   : {'PASSED (Valid)' if report['is_valid'] else 'FAILED'}")
    print("-" * 70)
    print(f"  Total Watchlist Records     : {report['total_records']}")
    print(f"  Active Records (Matchable)  : {report['active_records']}")
    print(f"  Inactive / Ineligible       : {report['inactive_records']}")
    print(f"  Duplicate Registrations     : {report['duplicate_count']}")
    print(f"  Invalid Formats             : {report['invalid_format_count']}")
    print(f"  Schema / Field Errors       : {report['schema_errors_count']}")
    print("-" * 70)
    print("  STEP 7 DEMONSTRATION COVERAGE:")
    print(f"  Guaranteed Matching Vehicles: {report['observed_counterparts_count']}")
    for m in report["guaranteed_matches"]:
        rec = repo.get_vehicle(m)
        print(f"    - {m:<14} [{rec.category if rec else 'N/A'}] (Priority: {rec.priority if rec else 'N/A'})")
    print(f"  Unobserved Counterparts     : {report['unobserved_counterparts_count']} (Non-matching controls)")
    for nm in report["non_matching_records"]:
        rec = repo.get_vehicle(nm)
        print(f"    - {nm:<14} [{rec.category if rec else 'N/A'}] (Status: {rec.status if rec else 'N/A'})")
    print(f"  Coverage Ratio              : {report['observed_coverage_ratio']}")
    print("=" * 70 + "\n")

    if not report["is_valid"]:
        if report["duplicates"]:
            print(f"[!] Duplicate normalized registrations found: {report['duplicates']}")
        if report["schema_errors"]:
            print(f"[!] Schema errors detected: {report['schema_errors']}")
        sys.exit(1)
    else:
        print("[+] Watchlist integrity verified successfully. Ready for Step 9 matching.")
        sys.exit(0)


if __name__ == "__main__":
    main()
