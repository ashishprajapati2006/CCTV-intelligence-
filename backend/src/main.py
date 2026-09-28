"""Gujarat Police CCTV Platform - Foundation CLI Entrypoint."""
import sys
import argparse
from pathlib import Path

from src.catalogue.sync import sync_catalogue
from src.streaming.rtsp import RTSPStreamReader
from scripts.test_catalogue_access import run_diagnostic
from scripts.list_cameras import main as list_cameras_main
from scripts.test_camera import run_stream_test


def main():
    parser = argparse.ArgumentParser(
        description="Gujarat Police CCTV Integration Platform - Phase 1-3 Controller",
    )
    subparsers = parser.add_subparsers(dest="command", help="Platform actions")

    # Command: diagnose
    subparsers.add_parser("diagnose", help="Run Sentinel catalogue connectivity diagnostics")

    # Command: sync
    sync_p = subparsers.add_parser("sync", help="Synchronize camera catalogue")
    sync_p.add_argument("--source", choices=["live", "local"], default="live")
    sync_p.add_argument("--file", dest="local_file", default=None)

    # Command: list
    subparsers.add_parser("list", help="List registered cameras")

    # Command: stream
    stream_p = subparsers.add_parser("stream", help="Validate camera RTSP stream")
    stream_p.add_argument("--camera-id", default="cam01", help="Camera ID to stream")
    stream_p.add_argument("--frames", type=int, default=100, help="Frame count limit (default: 100)")
    stream_p.add_argument("--headless", action="store_true", help="Run headless without window preview")

    args = parser.parse_args()

    if args.command == "diagnose":
        run_diagnostic()
    elif args.command == "sync":
        sync_catalogue(source=args.source, local_file=args.local_file)
    elif args.command == "list":
        list_cameras_main()
    elif args.command == "stream":
        run_stream_test(camera_id=args.camera_id, max_frames=args.frames, headless=args.headless)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
