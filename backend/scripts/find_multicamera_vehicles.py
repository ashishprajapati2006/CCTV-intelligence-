#!/usr/bin/env python3
"""find_multicamera_vehicles.py — Step 11.1 (Part 16)

Reads observations.jsonl and finds vehicles observed on 2+ distinct cameras.
Ranks by: number of cameras, total observations, average consensus score.

Usage:
    python scripts/find_multicamera_vehicles.py [--include-probable] [--min-cameras N]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_observations(obs_path: Path, include_probable: bool = False):
    """Load observation records from JSONL file."""
    records = []
    if not obs_path.exists():
        print(f"[ERROR] Observations file not found: {obs_path}")
        return records

    allowed = {"CONFIRMED"}
    if include_probable:
        allowed.add("PROBABLE")

    with open(obs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                status = rec.get("status") or rec.get("recognition_status")
                if status in allowed:
                    rec["status"] = status
                    records.append(rec)
            except json.JSONDecodeError:
                pass
    return records


def find_multicamera_vehicles(
    obs_path: Path = Path("data/observed/vehicles/observations.jsonl"),
    include_probable: bool = False,
    min_cameras: int = 2,
) -> list:
    """Return list of vehicle summaries seen on >= min_cameras distinct cameras."""
    records = load_observations(obs_path, include_probable)

    # Group by registration
    by_reg: dict = defaultdict(list)
    for rec in records:
        reg = rec.get("normalized_registration_number") or rec.get("registration_number", "")
        if reg:
            by_reg[reg].append(rec)

    results = []
    for reg, obs_list in by_reg.items():
        cameras = list({r.get("camera_id") for r in obs_list})
        if len(cameras) < min_cameras:
            continue

        consensus_scores = [
            float(r.get("consensus_score", 0.0))
            for r in obs_list
            if r.get("consensus_score") is not None
        ]
        avg_consensus = round(sum(consensus_scores) / len(consensus_scores), 4) if consensus_scores else 0.0

        confirmed = sum(1 for r in obs_list if r.get("status") == "CONFIRMED")
        probable = sum(1 for r in obs_list if r.get("status") == "PROBABLE")

        results.append({
            "registration_number": reg,
            "cameras_count": len(cameras),
            "cameras": sorted(cameras),
            "total_observations": len(obs_list),
            "confirmed_count": confirmed,
            "probable_count": probable,
            "avg_consensus_score": avg_consensus,
        })

    # Rank: most cameras first, then most observations, then highest consensus
    results.sort(key=lambda x: (-x["cameras_count"], -x["total_observations"], -x["avg_consensus_score"]))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Find vehicles observed on multiple cameras."
    )
    parser.add_argument(
        "--include-probable",
        action="store_true",
        help="Include PROBABLE status observations (default: CONFIRMED only)",
    )
    parser.add_argument(
        "--min-cameras",
        type=int,
        default=2,
        help="Minimum number of distinct cameras (default: 2)",
    )
    parser.add_argument(
        "--obs-file",
        default="data/observed/vehicles/observations.jsonl",
        help="Path to observations JSONL file",
    )
    args = parser.parse_args()

    results = find_multicamera_vehicles(
        obs_path=Path(args.obs_file),
        include_probable=args.include_probable,
        min_cameras=args.min_cameras,
    )

    print()
    print("=" * 60)
    print("MULTI-CAMERA VEHICLE FINDER — Step 11.1")
    print("=" * 60)
    print(f"  Min cameras threshold : {args.min_cameras}")
    print(f"  Include probable      : {args.include_probable}")
    print(f"  Source file           : {args.obs_file}")
    print()

    if not results:
        print("  No vehicles found on multiple cameras.")
        print()
        print("  This is expected if the observation dataset is small.")
        print("  Run the ANPR pipeline against more cameras to expand the dataset.")
        return

    print(f"  Multi-camera vehicles found: {len(results)}")
    print()

    for i, v in enumerate(results, 1):
        print(f"  {i}. {v['registration_number']}")
        print(f"       Cameras        : {v['cameras_count']}  {v['cameras']}")
        print(f"       Observations   : {v['total_observations']}  "
              f"(confirmed={v['confirmed_count']}, probable={v['probable_count']})")
        print(f"       Avg Consensus  : {v['avg_consensus_score']}")
        print()


if __name__ == "__main__":
    main()
