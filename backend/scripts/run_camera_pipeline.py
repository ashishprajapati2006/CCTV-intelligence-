#!/usr/bin/env python3
"""Multi-Camera Pipeline Automation Script (Steps 4 -> 5 -> 6).

Runs the end-to-end AI perception pipeline for one or more cameras sequentially:
  - Step 4: Vehicle Detection & Tracking (YOLOv8 + ByteTrack/IoU)
  - Step 5: Best-Frame Selection & License Plate Detection
  - Step 6: ANPR/OCR & Multi-Frame Consensus (EasyOCR + Format Validation)

Usage examples:
  # Run a single camera:
  python scripts/run_camera_pipeline.py --cameras cam01 --frames 5000

  # Run multiple cameras in batch on GPU laptop:
  python scripts/run_camera_pipeline.py --cameras cam01 cam02 cam03 cam04 --frames 10000 --gpu

  # Skip detection if already run, only rerun plate detection and ANPR:
  python scripts/run_camera_pipeline.py --cameras cam01 --skip-detection
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def run_cmd(cmd: List[str], desc: str) -> bool:
    print(f"\n[PIPELINE] >>> Starting: {desc}")
    print(f"[PIPELINE] Command: {' '.join(cmd)}")
    t0 = time.time()
    try:
        ret = subprocess.run(cmd, check=True)
        elapsed = time.time() - t0
        print(f"[PIPELINE] [SUCCESS] Completed: {desc} in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        print(f"[PIPELINE] [ERROR] Failed: {desc} (code {e.returncode}) after {elapsed:.1f}s")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Automated multi-camera runner for Step 4 -> Step 5 -> Step 6"
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        required=True,
        help="List of camera IDs to process (e.g. --cameras cam01 cam02 cam03)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Number of frames to process per camera (omit for continuous)",
    )
    parser.add_argument(
        "--detector-interval",
        type=int,
        default=3,
        help="Run vehicle detector every N frames (default: 3)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.40,
        help="Vehicle detection confidence (default: 0.40)",
    )
    parser.add_argument(
        "--min-consensus",
        type=float,
        default=0.05,
        help="Consensus score for CONFIRMED status (default: 0.05)",
    )
    parser.add_argument(
        "--min-probable",
        type=float,
        default=0.03,
        help="Consensus score for PROBABLE status (default: 0.03)",
    )
    parser.add_argument(
        "--min-confirmed-frames",
        type=int,
        default=1,
        help="Minimum frames for confirmed status (default: 1)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Force GPU mode for OCR",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Force CPU mode for OCR",
    )
    parser.add_argument(
        "--skip-detection",
        action="store_true",
        help="Skip Step 4 (vehicle detection) if already run",
    )
    parser.add_argument(
        "--skip-plates",
        action="store_true",
        help="Skip Step 5 (plate detection) if already run",
    )
    parser.add_argument(
        "--plate-format",
        default="auto",
        choices=["indian", "european", "auto"],
        help="Plate format validator (default: auto for flexible multi-format validation)",
    )
    parser.add_argument(
        "--plate-confidence",
        type=float,
        default=0.10,
        help="Minimum plate detector confidence (default: 0.10)",
    )
    args = parser.parse_args()

    python_bin = sys.executable
    total_cameras = len(args.cameras)

    print("=" * 70)
    print("  GUJARAT POLICE CCTV -- BATCH CAMERA PIPELINE RUNNER")
    print("=" * 70)
    print(f"  Target cameras       : {', '.join(args.cameras)}")
    print(f"  Detector interval    : {args.detector_interval}")
    print(f"  Min consensus score  : {args.min_consensus}")
    print(f"  Min probable score   : {args.min_probable}")
    print(f"  Min confirmed frames : {args.min_confirmed_frames}")
    print(f"  Plate confidence     : {args.plate_confidence}")
    print(f"  Plate format         : {args.plate_format}")
    print(f"  Skip detection       : {args.skip_detection}")
    print(f"  Skip plates          : {args.skip_plates}")
    print(f"  Device mode          : {'GPU' if args.gpu else ('CPU' if args.cpu else 'AUTO')}")
    print("=" * 70 + "\n")

    for idx, cam_id in enumerate(args.cameras, 1):
        print(f"\n[{idx}/{total_cameras}] >>> PROCESSING CAMERA: {cam_id} <<<")

        # ── Step 4: Vehicle Detection ──
        if not args.skip_detection:
            cmd_step4 = [
                python_bin,
                "scripts/run_vehicle_detection.py",
                "--camera-id", cam_id,
                "--detector-interval", str(args.detector_interval),
                "--no-display",
            ]
            if args.gpu:
                cmd_step4.append("--gpu")
            elif args.cpu:
                cmd_step4.append("--cpu")

            ok = run_cmd(cmd_step4, f"Step 4 (Vehicle Detection) for {cam_id}")
            if not ok:
                print(f"[PIPELINE] Skipping remaining steps for {cam_id} due to Step 4 failure.")
                continue

        # ── Step 5: Plate Detection ──
        if not args.skip_plates:
            cmd_step5 = [
                python_bin,
                "scripts/run_plate_detection.py",
                "--camera-id", cam_id,
                "--plate-confidence", str(args.plate_confidence),
                "--no-display",
            ]
            ok = run_cmd(cmd_step5, f"Step 5 (Plate Detection) for {cam_id}")
            if not ok:
                print(f"[PIPELINE] Skipping ANPR for {cam_id} due to Step 5 failure.")
                continue

        # ── Step 6: ANPR & Consensus ──
        cmd_step6 = [
            python_bin,
            "scripts/run_anpr.py",
            "--camera-id", cam_id,
            "--min-consensus", str(args.min_consensus),
            "--min-probable", str(args.min_probable),
            "--min-confirmed-frames", str(args.min_confirmed_frames),
            "--plate-format", args.plate_format,
            "--no-display",
        ]
        if args.gpu:
            cmd_step6.append("--gpu")
        elif args.cpu:
            cmd_step6.append("--cpu")

        run_cmd(cmd_step6, f"Step 6 (ANPR & Consensus) for {cam_id}")

    print("\n" + "=" * 70)
    print("  BATCH PIPELINE PROCESSING COMPLETE")
    print("=" * 70)
    print("Next step: Ingest results into Observed Vehicle DB (Step 7):")
    print("  python scripts/build_observed_vehicle_db.py --scan-dir data/anpr")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
