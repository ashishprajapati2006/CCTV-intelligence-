#!/usr/bin/env python3
"""report_observation_dataset.py — Step 11.1 (Parts 14, 25)

Generates data/observed/reports/dataset_statistics.json with full dataset metrics.
Also prints a human-readable summary to stdout.

Usage:
    python scripts/report_observation_dataset.py [--include-uncertain]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

OBS_FILE = Path("data/observed/vehicles/observations.jsonl")
UNCERTAIN_FILE = Path("data/observed/vehicles/uncertain_observations.jsonl")
OUTPUT_FILE = Path("data/observed/reports/dataset_statistics.json")


def count_jsonl(path: Path, include_uncertain: bool = False) -> dict:
    """Count records by status from a JSONL file."""
    counts = {"CONFIRMED": 0, "PROBABLE": 0, "UNCERTAIN": 0, "UNREADABLE": 0, "OTHER": 0}
    cameras = set()
    regs = set()
    total = 0

    if not path.exists():
        return {"counts": counts, "cameras": cameras, "regs": regs, "total": 0}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                status = rec.get("status", "OTHER")
                if status in counts:
                    counts[status] += 1
                else:
                    counts["OTHER"] += 1
                total += 1
                cam = rec.get("camera_id")
                if cam:
                    cameras.add(cam)
                reg = rec.get("normalized_registration_number") or rec.get("registration_number")
                if reg and status in {"CONFIRMED", "PROBABLE"}:
                    regs.add(reg)
            except json.JSONDecodeError:
                pass

    return {"counts": counts, "cameras": cameras, "regs": regs, "total": total}


def generate_dataset_statistics(include_uncertain: bool = False) -> dict:
    """Generate the full dataset statistics report."""
    main_stats = count_jsonl(OBS_FILE)
    unc_stats = count_jsonl(UNCERTAIN_FILE)

    # Multi-camera analysis from main observations
    by_reg: dict = defaultdict(set)
    if OBS_FILE.exists():
        with open(OBS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("status") in {"CONFIRMED", "PROBABLE"}:
                        reg = rec.get("normalized_registration_number") or rec.get("registration_number", "")
                        cam = rec.get("camera_id", "")
                        if reg and cam:
                            by_reg[reg].add(cam)
                except json.JSONDecodeError:
                    pass

    multi_camera_vehicles = [r for r, cams in by_reg.items() if len(cams) >= 2]
    all_cameras = main_stats["cameras"] | unc_stats["cameras"] if include_uncertain else main_stats["cameras"]

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": {
            "observations": str(OBS_FILE),
            "uncertain_observations": str(UNCERTAIN_FILE),
        },
        "observations_summary": {
            "unique_vehicles": len(main_stats["regs"]),
            "total_observations": main_stats["total"],
            "cameras_used": sorted(list(main_stats["cameras"])),
            "cameras_used_count": len(main_stats["cameras"]),
            "multi_camera_vehicles": len(multi_camera_vehicles),
            "multi_camera_vehicle_registrations": sorted(multi_camera_vehicles),
            "confirmed_count": main_stats["counts"]["CONFIRMED"],
            "probable_count": main_stats["counts"]["PROBABLE"],
        },
        "uncertain_observations_summary": {
            "total": unc_stats["total"],
            "uncertain_count": unc_stats["counts"]["UNCERTAIN"],
            "unreadable_count": unc_stats["counts"]["UNREADABLE"],
            "cameras_seen": sorted(list(unc_stats["cameras"])),
        },
        "coverage": {
            "sentinel_cameras_total": 30,
            "cameras_with_any_observation": len(all_cameras),
            "cameras_with_confirmed_or_probable": len(main_stats["cameras"]),
            "coverage_percent": round(len(main_stats["cameras"]) / 30 * 100, 1),
        },
        "data_quality": {
            "all_source_times_resolved": False,
            "source_time_anchor_available": False,
            "all_coordinates_available": False,
            "coordinate_source": "NOT_AVAILABLE",
            "limitations": [
                "Source/calendar time is NOT_RESOLVED for all observations — no time anchor in Sentinel catalogue",
                "Camera coordinates are unavailable — spatial analysis not possible",
                "Dataset covers only cameras that have been ingested through the ANPR pipeline",
            ],
        },
        "expansion_instructions": (
            "To expand the dataset, run: python scripts/run_anpr.py --camera <camera_id> "
            "then python scripts/build_observed_vehicle_db.py. "
            "DO NOT fabricate vehicle observations."
        ),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Generate observation dataset statistics report.")
    parser.add_argument("--include-uncertain", action="store_true",
                        help="Include uncertain/unreadable cameras in coverage stats")
    args = parser.parse_args()

    stats = generate_dataset_statistics(include_uncertain=args.include_uncertain)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Human-readable output
    obs = stats["observations_summary"]
    unc = stats["uncertain_observations_summary"]
    cov = stats["coverage"]
    dq = stats["data_quality"]

    print()
    print("=" * 60)
    print("OBSERVATION DATASET STATISTICS — Step 11.1")
    print("=" * 60)
    print()
    print("  CONFIRMED / PROBABLE OBSERVATIONS")
    print(f"    Unique vehicles          : {obs['unique_vehicles']}")
    print(f"    Total observations       : {obs['total_observations']}")
    print(f"    Confirmed count          : {obs['confirmed_count']}")
    print(f"    Probable count           : {obs['probable_count']}")
    print(f"    Cameras used             : {obs['cameras_used_count']}  {obs['cameras_used']}")
    print(f"    Multi-camera vehicles    : {obs['multi_camera_vehicles']}  {obs['multi_camera_vehicle_registrations']}")
    print()
    print("  UNCERTAIN / UNREADABLE")
    print(f"    Total                    : {unc['total']}")
    print(f"    Uncertain                : {unc['uncertain_count']}")
    print(f"    Unreadable               : {unc['unreadable_count']}")
    print(f"    Cameras seen             : {unc['cameras_seen']}")
    print()
    print("  COVERAGE")
    print(f"    Sentinel cameras total   : {cov['sentinel_cameras_total']}")
    print(f"    Cameras with any obs     : {cov['cameras_with_any_observation']}")
    print(f"    Coverage                 : {cov['coverage_percent']}%")
    print()
    print("  DATA QUALITY")
    for lim in dq["limitations"]:
        print(f"    [!] {lim}")
    print()
    print(f"  Report saved -> {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    main()
