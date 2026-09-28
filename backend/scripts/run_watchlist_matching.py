#!/usr/bin/env python3
"""Step 9 - Run Real-Time Watchlist Matching CLI.

Batch/Stream matching runner that ingests ANPR observations from Step 7
(data/observed/vehicles/observations.jsonl) and evaluates them against the
ACTIVE vehicle watchlist (data/watchlist/vehicles/watchlist.json).

Produces:
  data/matches/raw/matches.jsonl
  data/matches/confirmed/matches.jsonl
  data/matches/rejected/rejected.jsonl
  data/matches/deduplicated/dedup_summary.json

Usage:
  python scripts/run_watchlist_matching.py \
      --observations data/observed/vehicles/observations.jsonl \
      --watchlist data/watchlist/vehicles/watchlist.json
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
    MatchDecisionEngine,
    MatchEligibilityConfig,
)

logger = get_logger("run_matching")


def main():
    ap = argparse.ArgumentParser(description="Step 9 - Real-Time Watchlist Matching Runner")
    ap.add_argument("--observations", default="data/observed/vehicles/observations.jsonl",
                    help="Path to Step 7 observations.jsonl (default: data/observed/vehicles/observations.jsonl)")
    ap.add_argument("--watchlist", default="data/watchlist/vehicles/watchlist.json",
                    help="Path to vehicle watchlist (default: data/watchlist/vehicles/watchlist.json)")
    ap.add_argument("--output-dir", default="data/matches",
                    help="Output directory for match records (default: data/matches)")
    ap.add_argument("--suppression-window", type=float, default=30.0,
                    help="Deduplication suppression window in seconds / PTS (default: 30.0)")
    ap.add_argument("--fuzzy", action="store_true",
                    help="Enable possible-match review for single-character OCR discrepancies")
    ap.add_argument("--min-consensus", type=float, default=0.80,
                    help="Minimum consensus score threshold for eligibility (default: 0.80)")
    ap.add_argument("--min-ocr", type=float, default=0.70,
                    help="Minimum OCR confidence threshold for eligibility (default: 0.70)")
    args = ap.parse_args()

    obs_path = Path(args.observations)
    if not obs_path.exists():
        print(f"[ERROR] Observations file not found: {args.observations}")
        sys.exit(1)

    wl_path = Path(args.watchlist)
    if not wl_path.exists():
        print(f"[ERROR] Watchlist file not found: {args.watchlist}")
        sys.exit(1)

    print("=" * 70)
    print("  GUJARAT POLICE CCTV - REAL-TIME WATCHLIST MATCHING (STEP 9)")
    print("=" * 70)
    print(f"  Input Observations       : {args.observations}")
    print(f"  Target Watchlist         : {args.watchlist}")
    print(f"  Output Directory         : {args.output_dir}")
    print(f"  Deduplication Window     : {args.suppression_window} s")
    print(f"  Fuzzy Review Enabled     : {args.fuzzy}")
    print(f"  Eligibility Gate         : Consensus >= {args.min_consensus}, OCR >= {args.min_ocr}")
    print("=" * 70 + "\n")

    # Clear previous match logs if rebuilding
    out_dir = Path(args.output_dir)
    for sub in ("raw", "confirmed", "rejected", "deduplicated"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    for fname in (out_dir / "raw" / "matches.jsonl", out_dir / "confirmed" / "matches.jsonl", out_dir / "rejected" / "rejected.jsonl"):
        if fname.exists():
            fname.unlink()

    repo = JSONWatchlistRepository(vehicles_path=str(wl_path))
    elig_config = MatchEligibilityConfig(
        min_consensus_score=args.min_consensus,
        min_ocr_confidence=args.min_ocr,
    )
    engine = MatchDecisionEngine(
        repository=repo,
        eligibility_config=elig_config,
        enable_fuzzy_review=args.fuzzy,
        suppression_window_pts_ms=args.suppression_window * 1000.0,
        matches_dir=str(out_dir),
    )

    decisions = []
    with open(obs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obs = json.loads(line)
                dec = engine.process_observation(obs)
                decisions.append(dec)
            except Exception as e:
                logger.error("Failed to process observation line: %s", e)

    summary = engine.write_summary_report(decisions)

    print("=" * 70)
    print("  WATCHLIST MATCHING EXECUTION SUMMARY")
    print("=" * 70)
    print(f"  Total Observations Processed     : {summary['total_observations_processed']}")
    print(f"  Eligible Observations            : {summary['eligible_observations']}")
    print(f"  Ineligible Observations (Gated)  : {summary['not_eligible_observations']}")
    print("-" * 70)
    print(f"  Exact Watchlist Matches          : {summary['total_exact_matches']}")
    print(f"  --> Alert-Ready Matches          : {summary['alert_ready_matches']} (Passed deduplication)")
    print(f"  --> Suppressed Duplicates        : {summary['suppressed_duplicate_matches']} (Within time window)")
    print(f"  No Match (Unlisted / Inactive)   : {summary['no_matches']}")
    print(f"  Possible Match Reviews (Fuzzy)   : {summary['possible_match_reviews']}")
    print("-" * 70)
    print("  CONFIRMED ALERT-READY MATCHES:")
    for m in summary["alert_ready_events"]:
        meta = m.get("watchlist_metadata") or {}
        cat = meta.get("category", "N/A")
        prio = meta.get("priority", "N/A")
        pts_s = (m.get("recognition_pts_ms") or 0.0) / 1000.0
        print(f"    - [{m['camera_id']}] {m['normalized_registration_number']:<12} -> WL: {m['watchlist_id']} [{cat}] (Priority: {prio}, PTS: {pts_s:.3f} s)")
    print("=" * 70)
    print(f"\n[+] Match outputs written to {args.output_dir}/")


if __name__ == "__main__":
    main()
