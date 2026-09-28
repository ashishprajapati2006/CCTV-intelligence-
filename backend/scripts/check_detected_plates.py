#!/usr/bin/env python3
"""Inspect and report detected vehicle plates across all videos in Synthetic Dataset_result.

Reads anpr_consensus.json from each video result directory and displays:
  - Summary table of CONFIRMED, PROBABLE, UNCERTAIN, and UNREADABLE tracks.
  - Detailed listing of all CONFIRMED and PROBABLE plates.
  - (Optional) Partial / uncertain OCR text reads for debugging.

Usage:
  python scripts/check_detected_plates.py
  python scripts/check_detected_plates.py --show-uncertain
  python scripts/check_detected_plates.py --dir "Synthetic Dataset_result/anpr"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def inspect_anpr_results(anpr_base_dir: str, show_uncertain: bool = False) -> None:
    base_path = Path(anpr_base_dir)
    if not base_path.exists():
        print(f"[ERROR] Directory not found: {base_path}")
        sys.exit(1)

    # Discover video directories (e.g., '1', '2', ..., '8')
    subdirs = [p for p in base_path.iterdir() if p.is_dir()]

    def _sort_key(p: Path):
        try:
            return (0, int(p.name))
        except ValueError:
            return (1, p.name)

    video_dirs = sorted(subdirs, key=_sort_key)

    if not video_dirs:
        print(f"[WARN] No video folders found under: {base_path}")
        return

    print("=" * 75)
    print("  GUJARAT POLICE CCTV -- ANPR DETECTION REPORT (SYNTHETIC / LOCAL VIDEOS)")
    print("=" * 75)
    print(f"  Target Directory : {base_path.resolve()}")
    print(f"  Videos Evaluated : {len(video_dirs)}")
    print("=" * 75)

    all_confirmed = []
    all_probable = []
    all_uncertain = []

    col_label = "Camera" if any("cam" in p.name.lower() for p in video_dirs) else "Video"

    # Table Header
    print(f"\n{col_label:<10} {'Tracks':<8} {'CONFIRMED':<12} {'PROBABLE':<10} {'UNCERTAIN':<11} {'UNREADABLE':<11}")
    print("-" * 67)

    for vdir in video_dirs:
        json_file = vdir / "consensus" / "anpr_consensus.json"
        if not json_file.exists():
            print(f"{vdir.name:<8} [NOT PROCESSED YET]")
            continue

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"{vdir.name:<8} [ERROR READING: {e}]")
            continue

        total_tracks = data.get("total_tracks_processed", 0)
        c_counts = data.get("recognition_counts", {})
        n_conf = c_counts.get("CONFIRMED", 0)
        n_prob = c_counts.get("PROBABLE", 0)
        n_unc = c_counts.get("UNCERTAIN", 0)
        n_unread = c_counts.get("UNREADABLE", 0)

        print(f"{vdir.name:<8} {total_tracks:<8} {n_conf:<12} {n_prob:<10} {n_unc:<11} {n_unread:<11}")

        # Collect tracks
        for t in data.get("tracks", []):
            st = t.get("recognition_status")
            if st == "CONFIRMED":
                all_confirmed.append((vdir.name, t))
            elif st == "PROBABLE":
                all_probable.append((vdir.name, t))
            elif st == "UNCERTAIN":
                all_uncertain.append((vdir.name, t))

    print("-" * 65)
    print(f"TOTALS across all processed videos:")
    print(f"  * CONFIRMED Plates : {len(all_confirmed)}")
    print(f"  * PROBABLE Plates  : {len(all_probable)}")
    print(f"  * UNCERTAIN Reads  : {len(all_uncertain)}")
    print("=" * 75)

    # 1. Detail CONFIRMED Plates
    if all_confirmed:
        print("\n" + "#" * 75)
        print("  [+] CONFIRMED PLATES (High Confidence & Valid Registration Syntax)")
        print("#" * 75)
        print(f"{col_label:<10} {'Track ID':<12} {'Registration No':<18} {'Score':<8} {'Frames':<8}")
        print("-" * 75)
        for vname, t in all_confirmed:
            reg = t.get("final_registration_number", "N/A")
            score = t.get("consensus_score", 0.0)
            fc = t.get("supporting_frame_count", 0)
            print(f"{vname:<10} {t.get('track_id'):<12} {reg:<18} {score:<8.2f} {fc:<8}")
    else:
        print("\n  [INFO] No CONFIRMED plates found yet.")

    # 2. Detail PROBABLE Plates
    if all_probable:
        print("\n" + "#" * 75)
        print("  [?] PROBABLE PLATES (Passed Format Check, Moderate Confidence)")
        print("#" * 75)
        print(f"{col_label:<10} {'Track ID':<12} {'Registration No':<18} {'Score':<8} {'Frames':<8}")
        print("-" * 75)
        for vname, t in all_probable:
            reg = t.get("final_registration_number", "N/A")
            score = t.get("consensus_score", 0.0)
            fc = t.get("supporting_frame_count", 0)
            print(f"{vname:<10} {t.get('track_id'):<12} {reg:<18} {score:<8.2f} {fc:<8}")

    # 3. Optional UNCERTAIN Reads
    if show_uncertain and all_uncertain:
        print("\n" + "#" * 75)
        print("  [!] UNCERTAIN READS (Detected Text Characters / Partial Reads)")
        print("      (These characters were read by OCR but did not match Indian plate regex)")
        print("#" * 75)
        print(f"{'Video':<8} {'Track ID':<12} {'Score':<8} {'Top Candidate OCR Strings'}")
        print("-" * 75)
        for vname, t in all_uncertain:
            cands = t.get("candidate_scores", {})
            cands_str = ", ".join(f"'{k}' ({v:.2f})" for k, v in sorted(cands.items(), key=lambda x: x[1], reverse=True)[:5])
            print(f"{vname:<8} {t.get('track_id'):<12} {t.get('consensus_score', 0.0):<8.2f} {cands_str}")

    print("\n" + "=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Check confirmed detected plates from ANPR consensus results")
    parser.add_argument(
        "--dir",
        default="Synthetic Dataset_result/anpr",
        help="Path to ANPR results directory (default: Synthetic Dataset_result/anpr)",
    )
    parser.add_argument(
        "--show-uncertain",
        action="store_true",
        help="Also display uncertain/partial OCR strings that didn't match full plate format",
    )
    args = parser.parse_args()
    inspect_anpr_results(args.dir, show_uncertain=args.show_uncertain)


if __name__ == "__main__":
    main()
