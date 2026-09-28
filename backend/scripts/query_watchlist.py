#!/usr/bin/env python3
"""Step 8 - Query Synthetic Watchlist CLI.

Queries the representative synthetic watchlist for vehicle records,
priority status, and category breakdowns.

Usage:
  # Query specific vehicle by registration:
  python scripts/query_watchlist.py --registration GJ01AB1234

  # List all watchlist records:
  python scripts/query_watchlist.py --list

  # Show watchlist statistics:
  python scripts/query_watchlist.py --stats

  # Filter by category:
  python scripts/query_watchlist.py --category DEMO_STOLEN_VEHICLE
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
    WatchlistQueries,
)


def display_watchlist_record(veh_dict: dict) -> None:
    print("=" * 60)
    print("  WATCHLIST RECORD")
    print("=" * 60)
    print(f"  Registration        : {veh_dict['registration_number']}")
    print(f"  Normalized Reg      : {veh_dict['normalized_registration_number']}")
    print(f"  Watchlist ID        : {veh_dict['watchlist_id']}")
    print("-" * 60)
    print(f"  Category            : {veh_dict['category']}")
    print(f"  Priority            : {veh_dict['priority']}")
    print(f"  Status              : {veh_dict['status']}")
    print(f"  Source              : {veh_dict['source']}")
    print(f"  Synthetic           : {str(veh_dict['synthetic']).lower()}")
    print("-" * 60)
    print(f"  Description         : {veh_dict.get('description')}")
    if veh_dict.get("notes"):
        print(f"  Notes               : {veh_dict.get('notes')}")
    print(f"  Created At          : {veh_dict.get('created_at')}")
    print("=" * 60 + "\n")


def display_stats(records: list) -> None:
    cat_counts = {}
    prio_counts = {}
    stat_counts = {}

    for r in records:
        c = r.category
        p = r.priority
        s = r.status
        cat_counts[c] = cat_counts.get(c, 0) + 1
        prio_counts[p] = prio_counts.get(p, 0) + 1
        stat_counts[s] = stat_counts.get(s, 0) + 1

    print("=" * 60)
    print("  SYNTHETIC WATCHLIST STATISTICS")
    print("=" * 60)
    print(f"  Total Watchlist Records : {len(records)}")
    print("-" * 60)
    print("  Records by Status:")
    for st, count in sorted(stat_counts.items()):
        print(f"    - {st:<18}: {count}")
    print("-" * 60)
    print("  Records by Category:")
    for cat, count in sorted(cat_counts.items()):
        print(f"    - {cat:<26}: {count}")
    print("-" * 60)
    print("  Records by Priority:")
    for pr, count in sorted(prio_counts.items()):
        print(f"    - {pr:<18}: {count}")
    print("=" * 60 + "\n")


def main():
    ap = argparse.ArgumentParser(description="Step 8 - Query Synthetic Watchlist")
    ap.add_argument("--watchlist", default="data/watchlist/vehicles/watchlist.json",
                    help="Path to vehicle watchlist (default: data/watchlist/vehicles/watchlist.json)")
    ap.add_argument("--registration", default=None,
                    help="Query vehicle by registration number (e.g. GJ01AB1234)")
    ap.add_argument("--category", default=None,
                    help="Filter by category (e.g. DEMO_STOLEN_VEHICLE)")
    ap.add_argument("--priority", default=None,
                    help="Filter by priority (e.g. HIGH, CRITICAL)")
    ap.add_argument("--active-only", action="store_true",
                    help="Only list ACTIVE records")
    ap.add_argument("--list", action="store_true",
                    help="List all watchlist records")
    ap.add_argument("--stats", action="store_true",
                    help="Display summary statistics")
    ap.add_argument("--json", action="store_true",
                    help="Output raw JSON")
    args = ap.parse_args()

    repo = JSONWatchlistRepository(vehicles_path=args.watchlist)
    service = WatchlistQueries(repository=repo)

    if args.registration:
        rec = service.get_watchlist_vehicle(args.registration)
        if not rec:
            print(f"[!] No watchlist record found for registration: {args.registration}")
            sys.exit(1)
        if args.json:
            print(json.dumps(rec.to_dict(), indent=2))
        else:
            display_watchlist_record(rec.to_dict())

    elif args.category:
        records = service.list_watchlist_by_category(args.category)
        if args.json:
            print(json.dumps([r.to_dict() for r in records], indent=2))
        else:
            print(f"Found {len(records)} record(s) in category '{args.category}':")
            for r in records:
                print(f"  - {r.normalized_registration_number:<14} [{r.priority}] ({r.status})")

    elif args.priority:
        records = service.list_watchlist_by_priority(args.priority)
        if args.json:
            print(json.dumps([r.to_dict() for r in records], indent=2))
        else:
            print(f"Found {len(records)} record(s) with priority '{args.priority}':")
            for r in records:
                print(f"  - {r.normalized_registration_number:<14} [{r.category}] ({r.status})")

    elif args.list or args.active_only:
        records = service.get_active_watchlist() if args.active_only else service.all_vehicles()
        if args.json:
            print(json.dumps([r.to_dict() for r in records], indent=2))
        else:
            print("=" * 76)
            print(f"{'WATCHLIST ID':<16} {'REGISTRATION':<16} {'CATEGORY':<24} {'PRIORITY':<10} {'STATUS'}")
            print("-" * 76)
            for r in records:
                print(f"{r.watchlist_id:<16} {r.normalized_registration_number:<16} {r.category:<24} {r.priority:<10} {r.status}")
            print("=" * 76)
            print(f"Total: {len(records)} record(s)\n")

    elif args.stats or (not args.registration and not args.category and not args.priority and not args.list):
        records = service.all_vehicles()
        if args.json:
            print(json.dumps({"total": len(records)}, indent=2))
        else:
            display_stats(records)


if __name__ == "__main__":
    main()
