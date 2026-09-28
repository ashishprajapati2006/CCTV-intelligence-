#!/usr/bin/env python3
"""Step 11 - Build Vehicle Journey Reconstruction CLI.

Builds cross-camera vehicle journeys from Step 7 Observed Vehicle Database:
  - Loads observations and enriches with camera catalogue metadata
  - Correlates same-camera frame detections into camera-track segments
  - Evaluates temporal ordering (SOURCE_TIME vs CAMERA_LOCAL)
  - Evaluates spatial distance, implied speed, and plausibility
  - Outputs machine-readable JSON and human-readable TXT reports

Usage:
  # Build journeys for all observed vehicles:
  python scripts/build_vehicle_journey.py --all

  # Build journey for a single vehicle:
  python scripts/build_vehicle_journey.py --registration GJ01AB1234
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.journey import (
    JSONJourneyRepository,
    ObservationLoader,
    PlausibilityState,
    VehicleJourneyBuilder,
)


def main():
    ap = argparse.ArgumentParser(description="Step 11 - Build Vehicle Journey Reconstruction")
    ap.add_argument("--registration", help="Specific vehicle registration to build journey for")
    ap.add_argument("--all", action="store_true", help="Build journeys for all observed vehicles")
    ap.add_argument("--include-probable", action="store_true", help="Include PROBABLE recognition observations")
    ap.add_argument("--observations", default="data/observed/vehicles/observations.jsonl",
                    help="Path to observations file (default: data/observed/vehicles/observations.jsonl)")
    ap.add_argument("--cameras", default="data/catalogue/normalized/cameras.json",
                    help="Path to cameras catalogue (default: data/catalogue/normalized/cameras.json)")
    args = ap.parse_args()

    if not args.registration and not args.all:
        print("[ERROR] Please specify either --registration <PLATE> or --all")
        sys.exit(1)

    loader = ObservationLoader(observations_file=args.observations, cameras_file=args.cameras)
    builder = VehicleJourneyBuilder(loader=loader)
    repo = JSONJourneyRepository()

    registrations = [args.registration] if args.registration else loader.list_all_observed_registrations()

    print("=" * 72)
    print("  GUJARAT POLICE CCTV - VEHICLE JOURNEY RECONSTRUCTION (STEP 11)")
    print("=" * 72)
    print(f"  Observations Database : {args.observations}")
    print(f"  Camera Catalogue      : {args.cameras}")
    print(f"  Target Vehicles       : {len(registrations)}")
    print(f"  Include Probable OCR  : {args.include_probable}")
    print("=" * 72 + "\n")

    t0 = time.time()
    metrics = {
        "vehicles_queried": len(registrations),
        "observations_loaded": 0,
        "observations_correlated": 0,
        "journeys_generated": 0,
        "anomalous_legs": 0,
        "unresolved_time_observations": 0,
    }

    for reg in registrations:
        raw_obs = loader.load_observations_for_vehicle(reg, include_probable=args.include_probable)
        metrics["observations_loaded"] += len(raw_obs)
        for o in raw_obs:
            if o.source_time_status != "RESOLVED":
                metrics["unresolved_time_observations"] += 1

        journey = builder.build_from_observations(raw_obs, registration_number=reg)
        if journey and journey.segments:
            repo.save_journey(journey)
            metrics["journeys_generated"] += 1
            metrics["observations_correlated"] += len(journey.segments)
            for leg in journey.legs:
                if leg.plausibility == PlausibilityState.ANOMALOUS:
                    metrics["anomalous_legs"] += 1

            status_str = f"[{journey.status}] Mode: {journey.ordering_mode}"
            legs_str = f"{len(journey.legs)} leg(s)" if journey.legs else "No legs (single/unresolved)"
            print(f"  [+] {reg:<12} -> {len(journey.segments)} segments across {len(set(s.camera_id for s in journey.segments))} cameras | {legs_str} | {status_str}")

    elapsed_sec = time.time() - t0
    latency_ms = (elapsed_sec * 1000.0) / max(1, metrics["journeys_generated"])

    print("\n" + "=" * 72)
    print("  JOURNEY BUILD PERFORMANCE METRICS (Part 25)")
    print("=" * 72)
    print(f"  Vehicles Queried              : {metrics['vehicles_queried']}")
    print(f"  Observations Loaded           : {metrics['observations_loaded']}")
    print(f"  Observations Correlated       : {metrics['observations_correlated']}")
    print(f"  Journeys Generated            : {metrics['journeys_generated']}")
    print(f"  Total Processing Time         : {elapsed_sec * 1000.0:.2f} ms ({latency_ms:.2f} ms/journey)")
    print(f"  Anomalous Legs Detected       : {metrics['anomalous_legs']}")
    print(f"  Unresolved Time Observations  : {metrics['unresolved_time_observations']}")
    print(f"  Output Reports Directory      : data/journeys/reports/")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
