#!/usr/bin/env python3
"""Step 11.1 — Validate Vehicle Journey Reconstruction Integrity.

Audits journey consistency, time-basis adherence, coordinate validity,
evidence provenance, PTS consistency, and limitations disclosure.

Usage:
  python scripts/validate_journey.py --registration GJ01AB1234 [--include-probable]
  python scripts/validate_journey.py GJ01AB1234
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.journey import (
    ObservationLoader,
    OrderingMode,
    PlausibilityState,
    VehicleJourneyBuilder,
)
from src.watchlist.normalizer import normalize_watchlist_registration


def run_validation(registration: str, include_probable: bool = False) -> int:
    """Run all integrity checks on the journey for the given registration.

    Returns exit code: 0=all pass, 1=warnings, 2=failures, 3=not found.
    """
    norm_reg = normalize_watchlist_registration(registration)
    loader = ObservationLoader()
    builder = VehicleJourneyBuilder(loader=loader)

    raw_obs = loader.load_observations_for_vehicle(norm_reg, include_probable=include_probable)
    if not raw_obs:
        print(f"[!] No observations found for vehicle '{registration}'")
        return 3

    journey = builder.build_from_observations(raw_obs, registration_number=norm_reg)

    print()
    print("=" * 72)
    print(f"  JOURNEY INTEGRITY AUDIT — Step 11.1")
    print(f"  Vehicle : {norm_reg}")
    print("=" * 72)
    print()

    checks = []  # (status, description) — status: PASSED | WARNING | FAILED | INFO

    # ------------------------------------------------------------------
    # Check 1: Multi-camera separation preserved
    # ------------------------------------------------------------------
    distinct_cams = len(set(s.camera_id for s in journey.segments))
    if distinct_cams > 1:
        checks.append(("PASSED", f"Multi-camera separation: {distinct_cams} distinct cameras"))
    else:
        cam = journey.segments[0].camera_id if journey.segments else "none"
        checks.append(("INFO", f"Single-camera vehicle ({cam}) — no cross-camera legs"))

    # ------------------------------------------------------------------
    # Check 2: Common-Clock Rule (no cross-camera PTS arithmetic)
    # ------------------------------------------------------------------
    if journey.ordering_mode == OrderingMode.CAMERA_LOCAL.value:
        fabricated_delta = any(
            leg.time_delta_seconds is not None and leg.from_camera_id != leg.to_camera_id
            for leg in journey.legs
        )
        if not fabricated_delta:
            checks.append(("PASSED", "Common-Clock Rule: no cross-camera PTS time delta fabricated"))
        else:
            checks.append(("FAILED", "Common-Clock Rule VIOLATION: cross-camera time delta present without resolved time"))
    else:
        checks.append(("PASSED", "Source time RESOLVED — global ordering is valid"))

    # ------------------------------------------------------------------
    # Check 3: PTS consistency (first_seen <= recognition <= last_seen)
    # ------------------------------------------------------------------
    pts_violations = []
    for obs in raw_obs:
        f = obs.first_seen_pts_ms
        r = obs.recognition_pts_ms
        l_ = obs.last_seen_pts_ms
        if f is not None and r is not None and f > r:
            pts_violations.append(f"{obs.observation_id}: first_seen({f}) > recognition({r})")
        if r is not None and l_ is not None and r > l_:
            pts_violations.append(f"{obs.observation_id}: recognition({r}) > last_seen({l_})")
    if not pts_violations:
        checks.append(("PASSED", "PTS consistency: first_seen <= recognition <= last_seen on all observations"))
    else:
        for v in pts_violations:
            checks.append(("FAILED", f"PTS violation: {v}"))

    # ------------------------------------------------------------------
    # Check 4: Duplicate observation IDs
    # ------------------------------------------------------------------
    obs_ids = [o.observation_id for o in raw_obs]
    seen_ids = set()
    dupes = []
    for oid in obs_ids:
        if oid in seen_ids:
            dupes.append(oid)
        seen_ids.add(oid)
    if not dupes:
        checks.append(("PASSED", f"No duplicate observation IDs ({len(obs_ids)} checked)"))
    else:
        checks.append(("FAILED", f"Duplicate observation IDs found: {dupes}"))

    # ------------------------------------------------------------------
    # Check 5: Source time transparency (must not be silently upgraded)
    # ------------------------------------------------------------------
    resolved = [o for o in raw_obs if o.source_time_status == "RESOLVED"]
    not_resolved = [o for o in raw_obs if o.source_time_status == "NOT_RESOLVED"]
    if journey.ordering_mode == OrderingMode.SOURCE_TIME.value and not_resolved:
        checks.append(("FAILED", f"Ordering mode is SOURCE_TIME but {len(not_resolved)} obs have NOT_RESOLVED source time"))
    else:
        checks.append((
            "PASSED",
            f"Source time status transparent: {len(resolved)} RESOLVED, {len(not_resolved)} NOT_RESOLVED "
            f"-> ordering mode={journey.ordering_mode}"
        ))

    # ------------------------------------------------------------------
    # Check 6: time_resolution field present and accurate
    # ------------------------------------------------------------------
    if hasattr(journey, "time_resolution"):
        checks.append(("PASSED", f"time_resolution field present: {journey.time_resolution}"))
    else:
        checks.append(("FAILED", "time_resolution field missing from VehicleJourney"))

    # ------------------------------------------------------------------
    # Check 7: spatial_resolution field and coordinate non-fabrication
    # ------------------------------------------------------------------
    coords_count = sum(1 for s in journey.segments if s.camera_latitude is not None)
    if hasattr(journey, "spatial_resolution"):
        if coords_count == 0:
            if journey.spatial_resolution == "UNAVAILABLE":
                checks.append(("PASSED", "spatial_resolution=UNAVAILABLE correctly set (no coordinates in catalogue)"))
            else:
                checks.append(("FAILED", f"spatial_resolution={journey.spatial_resolution} but no coordinates present"))
        else:
            checks.append(("INFO", f"Coordinates present on {coords_count}/{len(journey.segments)} segments"))
    else:
        checks.append(("FAILED", "spatial_resolution field missing from VehicleJourney"))

    # ------------------------------------------------------------------
    # Check 8: distance_status on all legs
    # ------------------------------------------------------------------
    for leg in journey.legs:
        if not hasattr(leg, "distance_status"):
            checks.append(("FAILED", f"Leg {leg.leg_id} missing distance_status field"))
            break
        if leg.distance_status == "AVAILABLE" and leg.straight_line_distance_m is None:
            checks.append(("FAILED", f"Leg {leg.leg_id}: distance_status=AVAILABLE but straight_line_distance_m=None"))
    else:
        if journey.legs:
            ds_vals = {leg.distance_status for leg in journey.legs}
            checks.append(("PASSED", f"distance_status present on all {len(journey.legs)} leg(s): {ds_vals}"))
        else:
            checks.append(("INFO", "No legs (single-segment journey)"))

    # ------------------------------------------------------------------
    # Check 9: Evidence image paths
    # ------------------------------------------------------------------
    missing_evidence = [s.segment_id for s in journey.segments if not s.evidence_image]
    if not missing_evidence:
        checks.append(("PASSED", "All segments have evidence_image paths"))
    else:
        checks.append(("WARNING", f"Missing evidence_image on segments: {missing_evidence}"))

    # ------------------------------------------------------------------
    # Check 10: Anomalous legs
    # ------------------------------------------------------------------
    anomalous = [leg for leg in journey.legs if leg.plausibility == PlausibilityState.ANOMALOUS]
    if not anomalous:
        checks.append(("PASSED", f"No ANOMALOUS journey legs ({len(journey.legs)} total)"))
    else:
        checks.append(("WARNING", f"{len(anomalous)}/{len(journey.legs)} leg(s) ANOMALOUS"))

    # ------------------------------------------------------------------
    # Check 11: limitations list non-empty when time/coords unresolved
    # ------------------------------------------------------------------
    if hasattr(journey, "limitations"):
        if journey.time_resolution == "NOT_RESOLVED" and not journey.limitations:
            checks.append(("FAILED", "limitations list is empty despite time_resolution=NOT_RESOLVED"))
        elif journey.limitations:
            checks.append(("PASSED", f"limitations list populated ({len(journey.limitations)} entries)"))
        else:
            checks.append(("INFO", "limitations list empty (fully resolved journey)"))
    else:
        checks.append(("FAILED", "limitations field missing from VehicleJourney"))

    # ------------------------------------------------------------------
    # Check 12: confidence score present and within 0-1
    # ------------------------------------------------------------------
    if journey.confidence_score:
        score = journey.confidence_score.overall_score
        if 0.0 <= score <= 1.0:
            checks.append(("PASSED", f"Confidence score in valid range: {score:.4f}"))
        else:
            checks.append(("FAILED", f"Confidence score out of range: {score}"))
        if hasattr(journey.confidence_score, "explanations") and journey.confidence_score.explanations:
            checks.append(("PASSED", f"explanations list populated ({len(journey.confidence_score.explanations)} items)"))
        else:
            checks.append(("WARNING", "explanations list is empty on confidence score"))
    else:
        checks.append(("WARNING", "No confidence score computed"))

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    passed = sum(1 for s, _ in checks if s == "PASSED")
    warnings = sum(1 for s, _ in checks if s == "WARNING")
    failures = sum(1 for s, _ in checks if s == "FAILED")

    for status, desc in checks:
        icon = {"PASSED": "+", "WARNING": "!", "FAILED": "X", "INFO": "i"}.get(status, " ")
        print(f"  {icon} [{status:<7}] {desc}")

    print()
    print("=" * 72)
    print(f"  RESULT: {passed} PASSED | {warnings} WARNINGS | {failures} FAILED")
    print("=" * 72)
    print()

    if failures > 0:
        return 2
    if warnings > 0:
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description="Validate vehicle journey integrity (Step 11.1)")
    # Support both positional and --registration for convenience
    parser.add_argument("registration_positional", nargs="?", help="Registration number (positional)")
    parser.add_argument("--registration", help="Registration number")
    parser.add_argument("--include-probable", action="store_true")
    args = parser.parse_args()

    reg = args.registration or args.registration_positional
    if not reg:
        parser.print_help()
        sys.exit(1)

    exit_code = run_validation(reg, include_probable=args.include_probable)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
