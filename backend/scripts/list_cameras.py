#!/usr/bin/env python3
"""Script to display normalized camera catalogue in a structured table.

Usage:
    python scripts/list_cameras.py [--file path/to/cameras.json]
"""
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.catalogue.models import Camera


def display_cameras(cameras: List[Camera]) -> None:
    """Print a clean tabular overview of cameras."""
    print("=" * 110)
    print("                      GUJARAT POLICE CCTV PLATFORM - CAMERA INVENTORY")
    print("=" * 110)
    print(f"{'ID':<10} | {'Name':<28} | {'Location':<20} | {'Status':<8} | {'Codec':<7} | {'Resolution':<12} | {'Streams'}")
    print("-" * 110)

    if not cameras:
        print("  No cameras found in normalized catalogue.")
        print("=" * 110)
        return

    for cam in cameras:
        cam_id = (cam.camera_id or "")[:10]
        name = (cam.name or "Unnamed")[:28]
        location = (cam.location or "Unknown")[:20]
        status = (cam.status or "unknown")[:8]
        codec = (cam.codec or "UNKNOWN")[:7]
        resolution = cam.resolution_str[:12]

        streams = []
        if cam.has_rtsp:
            streams.append("RTSP")
        if cam.has_hls:
            streams.append("HLS")
        if cam.webrtc_url:
            streams.append("WHEP")
        stream_str = "/".join(streams) if streams else "None"

        print(f"{cam_id:<10} | {name:<28} | {location:<20} | {status:<8} | {codec:<7} | {resolution:<12} | {stream_str}")

    print("=" * 110)
    print(f"Total registered cameras: {len(cameras)}\n")


def main():
    parser = argparse.ArgumentParser(description="List normalized cameras from catalogue.")
    parser.add_argument(
        "--file",
        default="data/catalogue/normalized/cameras.json",
        help="Path to normalized cameras.json (default: data/catalogue/normalized/cameras.json)",
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"[-] Error: Catalogue file not found at '{file_path}'.")
        print("    Run 'python -m src.catalogue.sync' first to populate the catalogue.")
        sys.exit(1)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cameras = [Camera.from_dict(item) for item in data]
        display_cameras(cameras)
    except Exception as e:
        print(f"[-] Error reading catalogue: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
