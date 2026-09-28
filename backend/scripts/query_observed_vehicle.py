#!/usr/bin/env python3
"""Step 7 - Query Observed Vehicle Database CLI.

Queries the CCTV Observed Vehicle Database for vehicle identity profiles,
cross-camera observation histories, sighting sequences, track evidence,
and network-wide statistics.

Enforces strict timestamp semantics:
  - Distinguishes Media PTS from source calendar time and application ingestion time.
  - Formats cross-camera observations as Media Timeline / Observation Sequences,
    without asserting speculative chronological routes across independent camera clocks.

Usage:
  # Query specific vehicle by registration number:
  python scripts/query_observed_vehicle.py --registration GJ01AB1234

  # List all observed vehicles:
  python scripts/query_observed_vehicle.py --list

  # Show network-wide CCTV observation statistics:
  python scripts/query_observed_vehicle.py --stats

  # Inspect a specific vehicle track:
  python scripts/query_observed_vehicle.py --track TRK-0001
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.observed import (
    JSONObservedVehicleRepository,
    ObservedVehicleQueries,
)


def _fmt_s(pts_ms: float | None) -> str:
    """Format milliseconds to seconds string with 3 decimal places."""
    if pts_ms is None:
        return "N/A"
    return f"{pts_ms / 1000.0:.3f} s ({pts_ms:.1f} ms)"


def display_vehicle(veh_dict: dict) -> None:
    print("=" * 68)
    print("  OBSERVED VEHICLE PROFILE")
    print("=" * 68)
    print(f"  Registration Number        : {veh_dict['registration_number']}")
    print(f"  Normalized Registration    : {veh_dict['normalized_registration_number']}")
    print(f"  Vehicle ID                 : {veh_dict['vehicle_id']}")
    print(f"  Database Status            : {veh_dict['status']}")
    print("-" * 68)
    fs = veh_dict.get("first_seen", {})
    ls = veh_dict.get("last_seen", {})
    fs_src = fs.get("source_time") or "NOT_RESOLVED"
    ls_src = ls.get("source_time") or "NOT_RESOLVED"
    print(f"  First Sighting             : Camera: {fs.get('camera_id')}, PTS: {_fmt_s(fs.get('pts_ms'))}")
    print(f"                               Source Calendar Time: {fs_src}")
    print(f"  Last Sighting              : Camera: {ls.get('camera_id')}, PTS: {_fmt_s(ls.get('pts_ms'))}")
    print(f"                               Source Calendar Time: {ls_src}")
    print("-" * 68)
    print(f"  Cross-Camera Sighting Count: {veh_dict['camera_count']} camera(s)")
    print("  Cameras Observed At:")
    for cam in veh_dict.get("cameras", []):
        print(f"    - {cam}")
    print("-" * 68)
    print(f"  Total Observations Count   : {veh_dict['observation_count']}")
    print(f"  Associated Tracks Count    : {veh_dict['track_count']}")
    print(f"  Best Consensus Score       : {veh_dict['best_consensus_score']:.4f}")
    print(f"  Average Consensus Score    : {veh_dict['average_consensus_score']:.4f}")
    print("-" * 68)
    print("  CROSS-CAMERA OBSERVATION HISTORY")
    print("  [Media Timeline / Observation Sequence -- Not verified chronological route]")
    print("-" * 68)
    for entry in veh_dict.get("timeline", []):
        cam = entry.get("camera_id")
        trk = entry.get("track_id")
        f_pts = entry.get("first_seen_pts_ms")
        r_pts = entry.get("recognition_pts_ms")
        l_pts = entry.get("last_seen_pts_ms")
        src_t = entry.get("source_time") or "NOT_RESOLVED"
        score = entry.get("consensus_score", 0.0)
        status = entry.get("status", "CONFIRMED")
        fn_status = entry.get("evidence_filename_pts_status", "UNKNOWN")

        print(f"  {cam} -> Track: {trk} [{status}] (consensus score: {score:.2f})")
        print(f"      first PTS       : {_fmt_s(f_pts)}")
        print(f"      recognition PTS : {_fmt_s(r_pts)}")
        print(f"      last PTS        : {_fmt_s(l_pts)}")
        print(f"      source time     : {src_t}")
        if entry.get("evidence_image"):
            print(f"      evidence image  : {entry.get('evidence_image')} (PTS audit: {fn_status})")
        print(f"      ingested at UTC : {entry.get('ingested_at_utc')}")
        print()
    print("=" * 68 + "\n")


def display_track(trk_dict: dict) -> None:
    print("=" * 68)
    print("  VEHICLE TRACK RECORD")
    print("=" * 68)
    print(f"  Track ID                   : {trk_dict['track_id']}")
    print(f"  Camera ID                  : {trk_dict['camera_id']}")
    print(f"  Registration Number        : {trk_dict.get('registration_number') or 'N/A'}")
    print(f"  Recognition Status         : {trk_dict['status']}")
    print(f"  Consensus Score            : {trk_dict['consensus_score']:.4f}")
    print(f"  First Seen PTS             : {_fmt_s(trk_dict.get('first_seen_pts_ms'))}")
    print(f"  Recognition PTS            : {_fmt_s(trk_dict.get('recognition_pts_ms'))}")
    print(f"  Last Seen PTS              : {_fmt_s(trk_dict.get('last_seen_pts_ms'))}")
    src_t = trk_dict.get("source_time") or "NOT_RESOLVED"
    print(f"  Source Calendar Time       : {src_t}")
    if trk_dict.get("loop_instance") is not None:
        print(f"  Loop Instance              : {trk_dict.get('loop_instance')}")
    print(f"  Ingested at UTC            : {trk_dict.get('ingested_at_utc')}")
    print(f"  Supporting Frames Count    : {len(trk_dict.get('supporting_frames', []))}")
    print("-" * 68)
    print("  Supporting Frame Evidence Chain:")
    for f in trk_dict.get("supporting_frames", []):
        pts_val = f.get("pts_ms")
        fn_status = f.get("filename_pts_status", "UNKNOWN")
        fn_pts = f.get("filename_pts_ms")
        fn_str = f" [filename pts: {fn_pts:.1f}ms - {fn_status}]" if fn_pts is not None else ""
        print(f"    Frame {f.get('frame_id'):<6} (evidence PTS: {_fmt_s(pts_val)}){fn_str}")
        if f.get("evidence_image"):
            print(f"      Image: {f.get('evidence_image')}")
    print("=" * 68 + "\n")


def display_stats(stats: dict) -> None:
    print("=" * 68)
    print("  OBSERVED VEHICLE DATABASE STATISTICS")
    print("=" * 68)
    print(f"  Total Unique Observed Vehicles : {stats['total_observed_vehicles']}")
    print(f"  Total Vehicle Observations     : {stats['total_observations']}")
    print(f"  Total Preserved Tracks         : {stats['total_tracks']}")
    print(f"  Uncertain / Unreadable Evidence: {stats['uncertain_observations_count']}")
    print(f"  Average Consensus Score        : {stats['average_consensus_score']:.4f}")
    print("-" * 68)
    print("  Observations by Camera:")
    for cam, count in sorted(stats.get("observations_by_camera", {}).items()):
        print(f"    - {cam:<16}: {count}")
    print("-" * 68)
    print("  Observations by Status:")
    for st, count in sorted(stats.get("observations_by_status", {}).items()):
        print(f"    - {st:<16}: {count}")
    print("=" * 68 + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Step 7 - Query Observed Vehicle Database"
    )
    ap.add_argument("--storage-dir", default="data/observed/vehicles",
                    help="Storage directory (default: data/observed/vehicles)")
    ap.add_argument("--registration", default=None,
                    help="Query vehicle by registration number (e.g. GJ01AB1234)")
    ap.add_argument("--track", default=None,
                    help="Query track evidence by track_id (e.g. TRK-0001)")
    ap.add_argument("--list", action="store_true",
                    help="List all observed vehicles")
    ap.add_argument("--stats", action="store_true",
                    help="Display network-wide observation statistics")
    ap.add_argument("--json", action="store_true",
                    help="Output raw JSON instead of formatted text")
    args = ap.parse_args()

    repo = JSONObservedVehicleRepository(storage_dir=args.storage_dir)
    service = ObservedVehicleQueries(repository=repo)

    if args.registration:
        veh = service.get_vehicle(args.registration)
        if not veh:
            print(f"[!] No observed vehicle found for registration: {args.registration}")
            sys.exit(1)
        if args.json:
            print(json.dumps(veh.to_dict(), indent=2))
        else:
            display_vehicle(veh.to_dict())

    elif args.track:
        trk = service.get_track(args.track)
        if not trk:
            print(f"[!] No track found with ID: {args.track}")
            sys.exit(1)
        if args.json:
            print(json.dumps(trk.to_dict(), indent=2))
        else:
            display_track(trk.to_dict())

    elif args.list:
        vehicles = service.list_observed_vehicles(limit=500)
        if args.json:
            print(json.dumps([v.to_dict() for v in vehicles], indent=2))
        else:
            print("=" * 75)
            print(f"{'VEHICLE ID':<14} {'REGISTRATION':<16} {'OBSERVATIONS':<14} {'CAMERAS':<12} {'SCORE'}")
            print("-" * 75)
            for v in vehicles:
                cams = ", ".join(v.cameras)
                print(f"{v.vehicle_id:<14} {v.registration_number:<16} {v.observation_count:<14} {cams:<12} {v.best_consensus_score:.2f}")
            print("=" * 75)
            print(f"Total: {len(vehicles)} observed vehicle(s)\n")

    elif args.stats or (not args.registration and not args.track and not args.list):
        stats = service.get_statistics()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            display_stats(stats)


if __name__ == "__main__":
    main()
