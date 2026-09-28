#!/usr/bin/env python3
r"""Batch process local MP4 videos through the AI pipeline.

This script scans a directory of local videos, creates a temporary camera catalogue
where each video is treated as an independent camera stream, and runs Steps 4, 5, and 6
on them sequentially. Results are saved directly into the target output folder.

Usage:
  python scripts/process_local_videos.py \
      --input-dir "C:/Users/ADRAJ/Downloads/CCTV Platform/Synthetic Dataset" \
      --output-dir "C:/Users/ADRAJ/Downloads/CCTV Platform/Synthetic Dataset_result" \
      --detector-interval 40 \
      --cpu
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd: list[str], desc: str) -> bool:
    print(f"\n[PIPELINE] >>> Starting: {desc}")
    print(f"[PIPELINE] Command: {' '.join(cmd)}")
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - t0
        print(f"[PIPELINE] [SUCCESS] Completed: {desc} in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        print(f"[PIPELINE] [ERROR] Failed: {desc} (code {e.returncode}) after {elapsed:.1f}s")
        return False


def main():
    parser = argparse.ArgumentParser(description="Process local videos through CCTV AI pipeline.")
    parser.add_argument("--input-dir", required=True, help="Path to directory containing .mp4 videos")
    parser.add_argument("--output-dir", required=True, help="Path to save results")
    parser.add_argument("--detector-interval", type=int, default=1, help="Run detector every N frames (default: 1)")
    parser.add_argument("--min-ocr-confidence", type=float, default=0.02, help="Min OCR confidence (default: 0.02)")
    parser.add_argument("--min-consensus", type=float, default=0.05, help="Min confirmed consensus (default: 0.05)")
    parser.add_argument("--min-probable", type=float, default=0.03, help="Min probable score (default: 0.03)")
    parser.add_argument("--plate-confidence", type=float, default=0.10, help="Min plate confidence (default: 0.10)")
    parser.add_argument("--plate-format", default="european", choices=["indian", "european", "auto"],
                        help="Plate format to validate against (default: european for synthetic footage)")
    parser.add_argument("--gpu", action="store_true", help="Force GPU mode for OCR")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode for OCR")
    parser.add_argument("--skip-detection", action="store_true", help="Skip Step 4 if vehicle detections already exist")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    if not in_dir.exists():
        print(f"[ERROR] Input directory not found: {in_dir}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Subdirectories for outputs
    det_dir = out_dir / "detections"
    snap_dir = out_dir / "snapshots"
    anpr_dir = out_dir / "anpr"
    
    det_dir.mkdir(exist_ok=True)
    snap_dir.mkdir(exist_ok=True)
    anpr_dir.mkdir(exist_ok=True)

    # 1. Discover videos
    all_video_files = list(in_dir.glob("*.mp4")) + list(in_dir.glob("*.avi")) + list(in_dir.glob("*.mkv"))
    if not all_video_files:
        print(f"[ERROR] No video files found in {in_dir}")
        sys.exit(1)

    def _sort_key(p: Path):
        try:
            return (0, int(p.stem))
        except ValueError:
            return (1, p.stem)

    videos = sorted(all_video_files, key=_sort_key)

    # 2. Build temporary catalogue
    catalogue_path = out_dir / "synthetic_cameras.json"
    cameras = []
    for i, vp in enumerate(videos, start=1):
        cam_id = vp.stem if vp.stem.isdigit() else str(i)
        cameras.append({
            "camera_id": cam_id,
            "name": vp.name,
            "rtsp_url": str(vp.absolute()),
            "codec": "H264"
        })

    with open(catalogue_path, "w", encoding="utf-8") as f:
        json.dump(cameras, f, indent=2)

    print("=" * 70)
    print("  GUJARAT POLICE CCTV -- LOCAL VIDEO PIPELINE PROCESSOR")
    print("=" * 70)
    print(f"  Input Directory   : {in_dir}")
    print(f"  Output Directory  : {out_dir}")
    print(f"  Videos Found      : {len(videos)}")
    print(f"  Catalogue Created : {catalogue_path}")
    print("=" * 70)

    python_bin = sys.executable

    # 3. Process each camera
    for cam in cameras:
        cam_id = cam["camera_id"]
        vname = cam["name"]
        print("\n" + "#" * 70)
        print(f"  PROCESSING VIDEO: {vname} (as {cam_id})")
        print("#" * 70)

        # Step 4: Vehicle Detection
        if not args.skip_detection:
            cmd4 = [
                python_bin, "scripts/run_vehicle_detection.py",
                "--camera-id", cam_id,
                "--catalogue", str(catalogue_path),
                "--detector-interval", str(args.detector_interval),
                "--detections-dir", str(det_dir),
                "--snapshots-dir", str(snap_dir),
                "--no-display"
            ]
            if not run_cmd(cmd4, f"Step 4 (Vehicle Detection) for {cam_id}"):
                continue

        # Step 5: Plate Detection
        cmd5 = [
            python_bin, "scripts/run_plate_detection.py",
            "--camera-id", cam_id,
            "--tracks", str(det_dir / cam_id / "tracks.json"),
            "--catalogue", str(catalogue_path),
            "--detections", str(det_dir),
            "--snapshots", str(snap_dir),
            "--plate-confidence", str(args.plate_confidence),
            "--no-display"
        ]
        if not run_cmd(cmd5, f"Step 5 (Plate Detection) for {cam_id}"):
            continue

        # Step 6: ANPR & Consensus
        cmd6 = [
            python_bin, "scripts/run_anpr.py",
            "--camera-id", cam_id,
            "--input", str(det_dir / cam_id / "plate_candidates.jsonl"),
            "--output", str(anpr_dir),
            "--min-ocr-confidence", str(args.min_ocr_confidence),
            "--min-consensus", str(args.min_consensus),
            "--min-probable", str(args.min_probable),
            "--plate-format", args.plate_format,
            "--no-display"
        ]
        if args.gpu:
            cmd6.append("--gpu")
        elif args.cpu:
            cmd6.append("--cpu")
            
        run_cmd(cmd6, f"Step 6 (ANPR & Consensus) for {cam_id}")

    print("\n" + "=" * 70)
    print("  LOCAL VIDEO PROCESSING COMPLETE")
    print("=" * 70)
    print("To merge these results into your Observed Vehicle Database, run:")
    print(f"  python scripts/build_observed_vehicle_db.py --scan-dir \"{anpr_dir}\"")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
