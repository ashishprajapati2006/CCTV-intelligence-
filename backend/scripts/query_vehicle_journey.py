#!/usr/bin/env python3
"""Step 11 - Query Vehicle Journey CLI.

Answers the designated investigative query:
  "Where and when was a particular recognized vehicle observed across the integrated CCTV network?"

Usage:
  python scripts/query_vehicle_journey.py --registration GJ01AB1234
  python scripts/query_vehicle_journey.py --registration GJ05CD5678 --include-probable
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.journey import (
    ObservationLoader,
    OrderingMode,
    VehicleJourneyBuilder,
)
from src.watchlist.normalizer import normalize_watchlist_registration


def main():
    ap = argparse.ArgumentParser(description="Step 11 - Query Vehicle Journey & Sighting History")
    ap.add_argument("--registration", required=True, help="Designated vehicle registration (e.g. GJ01AB1234)")
    ap.add_argument("--include-probable", action="store_true", help="Include PROBABLE recognition candidate observations")
    ap.add_argument("--observations", default="data/observed/vehicles/observations.jsonl",
                    help="Path to observations file (default: data/observed/vehicles/observations.jsonl)")
    ap.add_argument("--cameras", default="data/catalogue/normalized/cameras.json",
                    help="Path to cameras catalogue (default: data/catalogue/normalized/cameras.json)")
    args = ap.parse_args()

    norm_reg = normalize_watchlist_registration(args.registration)
    loader = ObservationLoader(observations_file=args.observations, cameras_file=args.cameras)
    builder = VehicleJourneyBuilder(loader=loader)

    journey = builder.build_journey_for_vehicle(norm_reg, include_probable=args.include_probable)

    if not journey or not journey.segments:
        print(f"\n[!] No qualifying observations found for vehicle: '{args.registration}' (normalized: '{norm_reg}').")
        if not args.include_probable:
            print("    (Tip: Use --include-probable to search for PROBABLE recognition candidates.)\n")
        sys.exit(0)

    # ════════════════════════════════════════════════════════════════════════
    # Terminal Display matching Part 16 & 17 specifications
    # ════════════════════════════════════════════════════════════════════════

    if journey.ordering_mode == OrderingMode.SOURCE_TIME.value:
        print("\n" + "=" * 70)
        print("VEHICLE JOURNEY / OBSERVATION HISTORY")
        print("=" * 70)
        print(f"Vehicle:        {journey.registration_number}")
        print(f"Vehicle ID:     {journey.vehicle_id}")
        print(f"Mode:           {journey.ordering_mode}")
        print(f"Status:         {journey.status}")
        print("-" * 70)
        print("Observations:")
        for idx, seg in enumerate(journey.segments, 1):
            cam_name = f" ({seg.camera_name})" if seg.camera_name else ""
            coords = f" [{seg.camera_latitude:.4f}, {seg.camera_longitude:.4f}]" if seg.camera_latitude else ""
            print(f"{idx}. {seg.source_time} — {seg.camera_id}{cam_name} — Track: {seg.track_id}{coords}")
            print(f"   Recognition Conf: {seg.ocr_confidence:.2f} | Consensus: {seg.consensus_score:.2f} | Status: {seg.recognition_status}")
            print(f"   Evidence Image  : {seg.evidence_image or 'None'}")

        if journey.legs:
            print("-" * 70)
            print("Legs:")
            for leg in journey.legs:
                from_desc = f"{leg.from_camera_id}" + (f" ({leg.from_camera_name})" if leg.from_camera_name else "")
                to_desc = f"{leg.to_camera_id}" + (f" ({leg.to_camera_name})" if leg.to_camera_name else "")
                print(f"  {from_desc} → {to_desc}")
                if leg.time_delta_seconds is not None:
                    print(f"    Time Delta:        {leg.time_delta_seconds:.0f} s ({leg.time_delta_seconds/60:.1f} mins)")
                if leg.straight_line_distance_m is not None:
                    print(f"    Distance:          {leg.straight_line_distance_m:.1f} m (approximate straight-line)")
                if leg.implied_speed_kmh is not None:
                    print(f"    Implied Speed:     {leg.implied_speed_kmh:.1f} km/h (straight-line indicator)")
                print(f"    Plausibility:      {leg.plausibility.value} ({leg.plausibility_reason})")

    else:
        # CAMERA_LOCAL mode (Part 17)
        print("\n" + "=" * 70)
        print("VEHICLE OBSERVATION HISTORY")
        print("=" * 70)
        print(f"Vehicle:        {journey.registration_number}")
        print(f"Vehicle ID:     {journey.vehicle_id}")
        print("Time basis:     CAMERA-LOCAL MEDIA PTS")
        print(f"Status:         {journey.status}")
        print("-" * 70)

        # Group by camera for display
        by_cam = {}
        for s in journey.segments:
            by_cam.setdefault(s.camera_id, []).append(s)

        for cam_id, segs in by_cam.items():
            cam_name = f" ({segs[0].camera_name})" if segs[0].camera_name else ""
            coords = f" [Coordinates: {segs[0].camera_latitude:.4f}, {segs[0].camera_longitude:.4f}]" if segs[0].camera_latitude else " [Coordinates: UNAVAILABLE]"
            print(f"{cam_id.upper()}{cam_name}{coords}")
            for seg in segs:
                first_s = (seg.first_seen_pts_ms or 0.0) / 1000.0
                rec_s = (seg.recognition_pts_ms or 0.0) / 1000.0
                last_s = (seg.last_seen_pts_ms or 0.0) / 1000.0
                print(f"    Track:           {seg.track_id}")
                print(f"    First PTS:       {first_s:.3f} s ({seg.first_seen_pts_ms or 0.0:.1f} ms)")
                print(f"    Recognition PTS: {rec_s:.3f} s ({seg.recognition_pts_ms or 0.0:.1f} ms)")
                print(f"    Last PTS:        {last_s:.3f} s ({seg.last_seen_pts_ms or 0.0:.1f} ms)")
                print(f"    Status / Score:  {seg.recognition_status} (Consensus: {seg.consensus_score:.2f}, OCR: {seg.ocr_confidence:.2f})")
                print(f"    Evidence Image:  {seg.evidence_image or 'None'}")
                print("")

        print("=" * 70)
        print("NOTICE:")
        print("Camera-local PTS values are not globally comparable across cameras.")
        print("This output represents an unverified camera-local observation sequence, NOT a validated route.")
        print("=" * 70)

    if journey.confidence_score:
        cs = journey.confidence_score
        print(f"\nEngineering Confidence Score: {cs.overall_score:.2f} / 1.00")
        print(f"  - Recognition Quality  : {cs.recognition_quality:.2f}")
        print(f"  - Temporal Resolution  : {cs.temporal_resolution:.2f}")
        print(f"  - Spatial Resolution   : {cs.spatial_resolution:.2f}")
        print(f"  - Plausibility Score   : {cs.plausibility_consistency:.2f}")
        for note in cs.notes:
            print(f"    * {note}")

    print("\nMachine-readable report saved to: data/journeys/reports/" + norm_reg + "_journey.json\n")


if __name__ == "__main__":
    main()
