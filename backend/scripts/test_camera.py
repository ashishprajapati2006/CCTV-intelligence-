#!/usr/bin/env python3
"""OpenCV RTSP stream test utility with PTS timing and exponential backoff.

Usage:
    python scripts/test_camera.py --camera-id cam01 --frames 100
    python scripts/test_camera.py --camera-id cam01
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
import cv2

from src.catalogue.models import Camera
from src.streaming.rtsp import RTSPStreamReader
from src.common.logging import get_logger

load_dotenv()
logger = get_logger("test_camera")


def load_camera_from_catalogue(camera_id: str, catalogue_path: str) -> Camera:
    """Load camera record from normalized catalogue."""
    path = Path(catalogue_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Normalized catalogue not found at: {path}. "
            "Please run 'python -m src.catalogue.sync' first."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        cam = Camera.from_dict(item)
        if cam.camera_id == camera_id:
            return cam

    available = [item.get("camera_id") or item.get("id") for item in data if isinstance(item, dict)]
    raise ValueError(
        f"Camera '{camera_id}' not found in normalized catalogue. "
        f"Available cameras: {', '.join(filter(None, available))}"
    )


def run_stream_test(
    camera_id: str,
    catalogue_path: str = "data/catalogue/normalized/cameras.json",
    max_frames: Optional[int] = None,
    headless: bool = False,
) -> bool:
    """Execute RTSP stream validation for the specified camera."""
    print("============================================================")
    print("            SENTINEL RTSP STREAM VALIDATION                 ")
    print("============================================================")

    # 1. Discover camera from normalized catalogue
    try:
        camera = load_camera_from_catalogue(camera_id, catalogue_path)
    except Exception as e:
        print(f"[-] Catalogue Lookup Failed: {e}")
        return False

    print(f"Camera ID:   {camera.camera_id}")
    print(f"Name:        {camera.name or 'Unnamed'}")
    print(f"Location:    {camera.location or 'Unknown'}")
    print(f"Codec:       {camera.codec}")
    print(f"RTSP URL:    {camera.rtsp_url or 'None'}")
    print("------------------------------------------------------------")

    if not camera.rtsp_url:
        print(f"[-] Error: Camera '{camera_id}' does not have an RTSP stream URL in catalogue.")
        return False

    # Force RTSP over TCP
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    reader = RTSPStreamReader(
        camera_id=camera.camera_id,
        rtsp_url=camera.rtsp_url,
        codec=camera.codec,
        force_tcp=True,
    )

    print("[*] Enforcing RTSP over TCP via OPENCV_FFMPEG_CAPTURE_OPTIONS.")
    print("[*] Timing is anchored on media container PTS (cap.get(CAP_PROP_POS_MSEC)).")
    print("[*] Local arrival timestamps are logged separately and never used for video timing.")
    if max_frames:
        print(f"[*] Target validation frame count: {max_frames} frames.")
    else:
        print("[*] Running in continuous mode. Press 'q' in video window to exit.")
    print("============================================================\n")

    window_name = f"Gujarat Police CCTV - {camera.camera_id}"
    window_created = False

    try:
        for frame, meta in reader.frames(max_frames=max_frames):
            seq = meta["frame_sequence"]
            pts_ms = meta["media_pts_ms"]
            rec_mono = meta["local_receive_monotonic"]
            w = meta["width"]
            h = meta["height"]

            pts_display = f"{pts_ms:.2f}ms" if pts_ms is not None else "N/A"

            # Print frame timing diagnostics
            if seq <= 10 or seq % 25 == 0 or (max_frames and seq == max_frames):
                print(
                    f"[{meta['camera_id']}] Frame #{seq:04d} | Shape: {w}x{h} | "
                    f"PTS: {pts_display:<10} | Local Receive Clock: {rec_mono:.3f}s"
                )

            # Display window unless headless
            if not headless:
                try:
                    # Overlay diagnostics on frame
                    overlay_text = f"{camera.camera_id} | #{seq} | PTS: {pts_display}"
                    cv2.putText(
                        frame,
                        overlay_text,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(window_name, frame)
                    window_created = True

                    # Check for 'q' key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == ord("Q"):
                        print("\n[!] User interrupted stream viewing (pressed 'q').")
                        reader.stop()
                        break
                except cv2.error as cv_err:
                    # Headless or display-less environment fallback
                    headless = True
                    logger.warning("GUI display unavailable (%s). Switching to headless mode.", cv_err)

    except KeyboardInterrupt:
        print("\n[!] Stream test interrupted by keyboard.")
    finally:
        # Guarantee resource release
        reader.release()
        if window_created:
            cv2.destroyAllWindows()

    print("\n============================================================")
    print("                STREAM HEALTH REPORT                        ")
    print("============================================================")
    print(reader.health.summary_str())
    print("============================================================\n")

    if max_frames and reader.health.frames_received >= max_frames:
        print(f"[+] STREAM VALIDATION PASSED ({reader.health.frames_received}/{max_frames} frames received successfully)\n")
        return True
    elif not max_frames and reader.health.frames_received > 0:
        print(f"[+] STREAM INGESTION SUCCESSFUL ({reader.health.frames_received} frames received)\n")
        return True
    else:
        print("[-] Stream validation could not capture required frames.\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Sentinel CCTV RTSP Stream with OpenCV.")
    parser.add_argument(
        "--camera-id",
        default=os.getenv("DEFAULT_CAMERA_ID", "cam01"),
        help="Camera ID to test (default from .env or 'cam01')",
    )
    parser.add_argument(
        "--catalogue",
        default="data/catalogue/normalized/cameras.json",
        help="Path to normalized catalogue JSON",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Number of frames to validate (e.g. 100). If omitted, streams continuously.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening GUI preview window.",
    )

    args = parser.parse_args()

    success = run_stream_test(
        camera_id=args.camera_id,
        catalogue_path=args.catalogue,
        max_frames=args.frames,
        headless=args.headless,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
