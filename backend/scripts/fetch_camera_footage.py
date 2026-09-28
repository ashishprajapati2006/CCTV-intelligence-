#!/usr/bin/env python3
"""Download camera footage directly from Sentinel Cloud.

Usage:
    python scripts/fetch_camera_footage.py --camera cam01 --duration 30
    python scripts/fetch_camera_footage.py --camera cam02 --duration 60 --output data/footage/cam02.mp4
    python scripts/fetch_camera_footage.py --all --duration 10
"""
import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from src.streaming.sentinel_hls import SentinelHLSReader
from src.common.logging import get_logger

load_dotenv()
logger = get_logger("fetch_footage")


def fetch_footage(camera_id: str, duration: int = 30, output: str = None) -> bool:
    output_path = output or f"data/footage/{camera_id}_{duration}s.mp4"
    print(f"[*] Downloading {duration}s footage for {camera_id}...")
    reader = SentinelHLSReader(camera_id)
    success = reader.download_clip(output_path=output_path, duration_seconds=duration)
    if success:
        print(f"[+] Saved footage to: {output_path} ({os.path.getsize(output_path):,} bytes)")
    else:
        print(f"[-] Failed to download footage for {camera_id}")
    return success


def main():
    parser = argparse.ArgumentParser(description="Fetch Sentinel CCTV camera footage clip.")
    parser.add_argument("--camera", default="cam01", help="Camera ID (e.g. cam01, cam02)")
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds (default: 30)")
    parser.add_argument("--output", default=None, help="Output MP4 file path")
    parser.add_argument("--all", action="store_true", help="Download footage for cam01 through cam05")
    args = parser.parse_args()

    if args.all:
        for cid in [f"cam{i:02d}" for i in range(1, 6)]:
            fetch_footage(cid, args.duration)
    else:
        fetch_footage(args.camera, args.duration, args.output)


if __name__ == "__main__":
    main()
