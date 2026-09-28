#!/usr/bin/env python3
"""Step 9 - Test Watchlist Match CLI (Single Event).

Demonstrates matching an individual ANPR recognition observation against
the active watchlist without needing Kafka or external services.

Usage:
  python scripts/test_watchlist_match.py --plate GJ01AB1234
  python scripts/test_watchlist_match.py --plate GJ05CD5678
  python scripts/test_watchlist_match.py --plate GJ99ZZ9999
  python scripts/test_watchlist_match.py --plate GJ01A81234 --fuzzy
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
    MatchEligibilityConfig,
    MatchEligibilityEvaluator,
    WatchlistMatcher,
    normalize_watchlist_registration,
)


def main():
    ap = argparse.ArgumentParser(description="Step 9 - Test Watchlist Match on Single Plate")
    ap.add_argument("--plate", required=True, help="License plate registration string (e.g. GJ01AB1234)")
    ap.add_argument("--watchlist", default="data/watchlist/vehicles/watchlist.json",
                    help="Path to vehicle watchlist (default: data/watchlist/vehicles/watchlist.json)")
    ap.add_argument("--camera", default="cam01", help="Camera ID (default: cam01)")
    ap.add_argument("--track", default="TRK-0001", help="Track ID (default: TRK-0001)")
    ap.add_argument("--pts", type=float, default=11200.0, help="PTS in milliseconds (default: 11200.0)")
    ap.add_argument("--status", default="CONFIRMED", help="Recognition status (default: CONFIRMED)")
    ap.add_argument("--consensus", type=float, default=0.94, help="Consensus score (default: 0.94)")
    ap.add_argument("--ocr-conf", type=float, default=0.92, help="OCR confidence (default: 0.92)")
    ap.add_argument("--fuzzy", action="store_true", help="Enable optional fuzzy review matching")
    ap.add_argument("--json", action="store_true", help="Output raw JSON result")
    args = ap.parse_args()

    repo = JSONWatchlistRepository(vehicles_path=args.watchlist)
    matcher = WatchlistMatcher(repository=repo, enable_fuzzy_review=args.fuzzy)
    evaluator = MatchEligibilityEvaluator(config=MatchEligibilityConfig(allow_fuzzy_candidates=args.fuzzy))

    norm_reg = normalize_watchlist_registration(args.plate)

    obs = {
        "camera_id": args.camera,
        "track_id": args.track,
        "registration_number": args.plate,
        "normalized_registration_number": norm_reg,
        "recognition_status": args.status,
        "consensus_score": args.consensus,
        "ocr_confidence": args.ocr_conf,
        "plate_detection_confidence": 0.95,
        "plate_quality_score": 0.88,
        "recognition_pts_ms": args.pts,
    }

    # 1. Eligibility Check
    is_eligible, elig_reason = evaluator.evaluate(obs)

    if not is_eligible:
        decision = "NOT_ELIGIBLE"
        reason = elig_reason
        match_score = 0.0
        matched_veh = None
        fuzzy_diff = None
    else:
        decision, reason, match_score, matched_veh, fuzzy_diff = matcher.match(norm_reg)

    result = {
        "raw_plate": args.plate,
        "normalized_plate": norm_reg,
        "decision": decision,
        "decision_reason": reason,
        "watchlist_match_score": match_score,
        "recognition_confidence": args.ocr_conf,
        "consensus_score": args.consensus,
        "camera_id": args.camera,
        "track_id": args.track,
        "pts_ms": args.pts,
        "matched_vehicle": matched_veh.to_dict() if matched_veh else None,
        "fuzzy_diff": fuzzy_diff,
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("=" * 65)
    print("  STEP 9 -- WATCHLIST MATCH EVALUATION")
    print("=" * 65)
    print(f"  Observed Plate        : {args.plate}")
    print(f"  Normalized Plate      : {norm_reg}")
    print(f"  Camera / Track / PTS  : {args.camera} / {args.track} / {args.pts / 1000.0:.3f} s")
    print(f"  ANPR Quality          : Consensus={args.consensus:.2f}, OCR={args.ocr_conf:.2f}, Status={args.status}")
    print("-" * 65)
    print(f"  MATCH DECISION        : {decision}")
    print(f"  Decision Reason       : {reason}")
    print(f"  Watchlist Match Score : {match_score:.2f} (Exact text equality)")
    print(f"  Underlying OCR Conf   : {args.ocr_conf:.2f}")

    if matched_veh:
        print("-" * 65)
        print("  MATCHED WATCHLIST PROFILE (ACTIVE):")
        print(f"    Watchlist ID        : {matched_veh.watchlist_id}")
        print(f"    Category            : {matched_veh.category}")
        print(f"    Priority Level      : {matched_veh.priority}")
        print(f"    Record Status       : {matched_veh.status}")
        print(f"    Source Tag          : {matched_veh.source} (Synthetic={matched_veh.synthetic})")
        print(f"    Description         : {matched_veh.description}")

    if fuzzy_diff:
        print("-" * 65)
        print("  POSSIBLE MATCH DIAGNOSTICS (REVIEW ONLY):")
        print(f"    Target Plate        : {fuzzy_diff['target_registration']}")
        print(f"    Edit Distance       : {fuzzy_diff['edit_distance']}")
        print(f"    Similarity Score    : {fuzzy_diff['similarity_score']:.2f}")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
