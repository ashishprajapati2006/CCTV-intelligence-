#!/usr/bin/env python3
"""Step 4 -- Vehicle Detection & Multi-Object Tracking.

Reads from one Sentinel RTSP camera (discovered from the normalized catalogue),
runs YOLOv8n vehicle detection + IoU tracker, writes JSONL + track summaries,
collects best-candidate snapshots, and optionally shows a live annotated window.

Usage examples:
    python scripts/run_vehicle_detection.py --camera-id cam01
    python scripts/run_vehicle_detection.py --camera-id cam01 --frames 500
    python scripts/run_vehicle_detection.py --camera-id cam01 --frames 500 --no-display
    python scripts/run_vehicle_detection.py --camera-id cam01 --confidence 0.40 \\
        --detector-interval 3 --best-frames 5 --no-display

TIMING PRINCIPLES (enforced throughout):
  - media_pts_ms   = cap.get(CAP_PROP_POS_MSEC) -- authoritative video timeline
  - local_receive_monotonic = time.monotonic()  -- pipeline diagnostics only
  - CAP_PROP_FPS is NEVER used for timing calculations
  - Processing FPS reported = actual wall-clock throughput, not stream FPS

SCOPE BOUNDARY -- THIS SCRIPT DOES NOT:
  - Perform ANPR or OCR
  - Match vehicles against a watchlist
  - Write to PostgreSQL / PostGIS
  - Produce Kafka events
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
from dotenv import load_dotenv

# Project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src.catalogue.models import Camera
from src.streaming.rtsp import RTSPStreamReader
from src.ai.detector import VehicleDetector
from src.ai.tracker import VehicleTracker
from src.ai.schemas import DetectionEvent
from src.ai.frame_selector import evaluate_frame, update_best_frames, save_best_frames
from src.common.logging import get_logger
from src.common.time import get_monotonic_time

import json as _json

logger = get_logger("vehicle_detection")

# -- colours per vehicle class ------------------------------------------------
CLASS_COLORS = {
    "car":        (0, 200, 0),
    "truck":      (0, 100, 255),
    "bus":        (255, 100, 0),
    "motorcycle": (255, 0, 200),
    "bicycle":    (0, 220, 220),
}
DEFAULT_COLOR = (200, 200, 0)


def _load_camera(camera_id: str, catalogue_path: str) -> Camera:
    with open(catalogue_path, "r", encoding="utf-8") as f:
        data = _json.load(f)
    for item in data:
        cam = Camera.from_dict(item)
        if cam.camera_id == camera_id:
            return cam
    available = [i.get("camera_id", i.get("id", "?")) for i in data]
    raise ValueError(f"Camera '{camera_id}' not in catalogue. Available: {available}")


def _draw_tracks(frame, active_tracks, pts_ms, detector_ran: bool):
    """Overlay bounding boxes, labels, and PTS on the frame."""
    for track in active_tracks:
        if not track.current_bbox or track.frame_count < 1:
            continue
        x1, y1, x2, y2 = track.current_bbox
        color = CLASS_COLORS.get(track.vehicle_class, DEFAULT_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label1 = f"{track.track_id} | {track.vehicle_class.upper()}"
        if track.confidences:
            label1 += f" | {track.confidences[-1]:.2f}"
        pts_display = f"PTS: {pts_ms:.0f} ms" if pts_ms is not None else "PTS: N/A"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.52, 1
        tw, th = cv2.getTextSize(label1, font, scale, thick)[0]
        bg_y = max(y1 - th - 6, 0)
        cv2.rectangle(frame, (x1, bg_y), (x1 + tw + 4, bg_y + th + 4), color, -1)
        cv2.putText(frame, label1, (x1 + 2, bg_y + th + 1), font, scale, (0, 0, 0), thick, cv2.LINE_AA)
        cv2.putText(frame, pts_display, (x1, y2 + 14), font, 0.44, color, 1, cv2.LINE_AA)

    # HUD: detector indicator
    hud = f"DETECTOR {'ON' if detector_ran else 'SKIP'} | Tracks: {len(active_tracks)}"
    if pts_ms is not None:
        hud += f" | PTS {pts_ms:.0f}ms"
    cv2.putText(frame, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 100), 1, cv2.LINE_AA)
    return frame


def run_detection(
    camera_id: str,
    catalogue_path: str = "data/catalogue/normalized/cameras.json",
    stream_url: Optional[str] = None,
    max_frames: Optional[int] = None,
    confidence: float = 0.40,
    detector_interval: int = 3,
    best_frames: int = 5,
    show_display: bool = True,
    model_name: str = "yolov8n.pt",
    detections_dir: str = "data/detections",
    snapshots_dir: str = "data/snapshots",
) -> None:
    # -- Catalogue discovery ---------------------------------------------------
    camera = _load_camera(camera_id, catalogue_path)
    target_stream_url = stream_url or camera.rtsp_url
    if not target_stream_url:
        logger.error("Camera '%s' has no stream or RTSP URL.", camera_id)
        sys.exit(1)

    print("=" * 65)
    print("  GUJARAT POLICE CCTV - VEHICLE DETECTION & TRACKING (STEP 4)")
    print("=" * 65)
    print(f"  Camera  : {camera.camera_id} - {camera.name or 'Unnamed'}")
    print(f"  Stream  : {target_stream_url}")
    print(f"  Model   : {model_name}  |  Confidence >= {confidence}")
    print(f"  Detect every {detector_interval} frame(s)  |  Best-frames: {best_frames}")
    print(f"  Max frames : {max_frames or 'continuous'}")
    print(f"  Display    : {'YES (press Q to quit)' if show_display else 'HEADLESS'}")
    print("=" * 65)
    print("  NOTE: No ANPR, watchlist, or alert logic runs in this phase.")
    print("=" * 65 + "\n")

    # -- Output directories ----------------------------------------------------
    det_dir = Path(detections_dir) / camera_id
    snap_dir = Path(snapshots_dir) / camera_id
    det_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = det_dir / "vehicle_detections.jsonl"
    summary_path = det_dir / "tracks.json"

    # -- Detector -------------------------------------------------------------
    detector = VehicleDetector(
        model_name=model_name,
        confidence_threshold=confidence,
    )

    # -- Tracker --------------------------------------------------------------
    tracker = VehicleTracker(iou_threshold=0.30, max_age=30, min_hits=2)

    # -- RTSP reader (Step 3 unchanged) ---------------------------------------
    reader = RTSPStreamReader(
        camera_id=camera.camera_id,
        rtsp_url=target_stream_url,
        codec=camera.codec,
        force_tcp=True,
    )

    # -- Metrics --------------------------------------------------------------
    frames_received = 0
    frames_detected = 0
    total_vehicles = 0
    total_inference_ms = 0.0
    processing_start = get_monotonic_time()
    last_pts_ms: Optional[float] = None
    stream_reconnect_count = 0
    dropped_frames = 0
    # Sliding window for actual processing throughput
    frame_times: deque = deque(maxlen=30)

    # Frame buffer for snapshot saving {seq: frame_img}
    frame_buffer: dict = {}
    BUFFER_MAX = 200

    # Output file handle
    jsonl_file = open(jsonl_path, "a", encoding="utf-8")

    window_name = f"Sentinel -- {camera_id} -- Vehicle Detection"
    window_created = False
    headless = not show_display

    prev_pts_ms: Optional[float] = None   # for discontinuity tracking

    try:
        for frame, meta in reader.frames(max_frames=max_frames):
            t_frame_start = get_monotonic_time()
            frames_received += 1

            pts_ms: Optional[float] = meta["media_pts_ms"]
            seq: int = meta["frame_sequence"]
            local_mono: float = meta["local_receive_monotonic"]

            # -- PTS discontinuity - reset tracker --------------------------
            pts_discontinuity = False
            if pts_ms is not None and prev_pts_ms is not None:
                delta = pts_ms - prev_pts_ms
                if delta < 0 or delta > 10_000:   # backward or >10 s gap
                    logger.warning(
                        "[%s] PTS discontinuity detected (%.0f-%.0f). "
                        "Resetting tracker.",
                        camera_id, prev_pts_ms, pts_ms,
                    )
                    tracker.reset()
                    pts_discontinuity = True
            prev_pts_ms = pts_ms

            # -- Detect every N frames ---------------------------------------
            run_detector = (seq % detector_interval == 0)
            active_tracks = []
            inference_ms = 0.0

            if run_detector:
                try:
                    detections, inference_ms = detector.detect(
                        frame,
                        media_pts_ms=pts_ms,
                        local_receive_monotonic=local_mono,
                        frame_sequence=seq,
                    )
                except Exception as e:
                    logger.warning("Detector exception on frame %d: %s", seq, e)
                    detections = []
                    dropped_frames += 1

                frames_detected += 1
                total_vehicles += len(detections)
                total_inference_ms += inference_ms

                # Tracker update
                active_tracks = tracker.update(detections, media_pts_ms=pts_ms)

                # Write detection events to JSONL
                for track in active_tracks:
                    if track.frame_count < 2:
                        continue   # skip tentative 1-hit tracks
                    track.camera_id = camera_id
                    event = DetectionEvent(
                        camera_id=camera_id,
                        track_id=track.track_id,
                        pts_ms=pts_ms,
                        vehicle_class=track.vehicle_class,
                        confidence=track.confidences[-1] if track.confidences else 0.0,
                        bbox=track.current_bbox,
                        frame_sequence=seq,
                        local_receive_monotonic=local_mono,
                    )
                    jsonl_file.write(json.dumps(event.to_dict()) + "\n")
                jsonl_file.flush()

                # Best-frame scoring & collection
                frame_buffer[seq] = frame.copy()
                if len(frame_buffer) > BUFFER_MAX:
                    oldest = min(frame_buffer.keys())
                    del frame_buffer[oldest]

                for track in tracker.confirmed_tracks():
                    track.camera_id = camera_id
                    candidate = evaluate_frame(
                        frame=frame,
                        bbox=track.current_bbox,
                        media_pts_ms=pts_ms,
                        local_receive_monotonic=local_mono,
                        frame_sequence=seq,
                    )
                    update_best_frames(track, candidate, max_best_frames=best_frames)
                    track.camera_id = camera_id
                    save_best_frames(track, frame_buffer, str(snap_dir.parent), camera_id)

            else:
                # Non-detector frames: still update tracker with empty detections
                active_tracks = tracker.update([], media_pts_ms=pts_ms)

            # -- Live display ------------------------------------------------
            if not headless:
                display_frame = frame.copy()
                display_frame = _draw_tracks(display_frame, active_tracks, pts_ms, run_detector)

                try:
                    cv2.imshow(window_name, display_frame)
                    window_created = True
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q")):
                        logger.info("User pressed Q -- stopping.")
                        reader.stop()
                        break
                except cv2.error:
                    headless = True
                    logger.warning("Display unavailable -- switching to headless.")

            # -- Per-frame timing (actual throughput) -------------------------
            frame_times.append(get_monotonic_time() - t_frame_start)

            # -- Progress log every 50 processed frames -----------------------
            if frames_received % 50 == 0:
                elapsed = get_monotonic_time() - processing_start
                proc_fps = frames_received / elapsed if elapsed > 0 else 0.0
                avg_inf = total_inference_ms / max(frames_detected, 1)
                logger.info(
                    "[%s] Frames: %d | Detected: %d | Vehicles: %d | "
                    "Active Tracks: %d | Inf: %.1fms | Proc FPS: %.1f",
                    camera_id, frames_received, frames_detected, total_vehicles,
                    tracker.active_count(), avg_inf, proc_fps,
                )

    except KeyboardInterrupt:
        logger.info("Interrupted by keyboard.")
    finally:
        jsonl_file.close()
        reader.release()
        if window_created:
            cv2.destroyAllWindows()

    # -- Save best-frame snapshots ---------------------------------------------
    logger.info("Saving best-frame snapshots-")
    for track in tracker.all_tracks():
        track.camera_id = camera_id
        save_best_frames(track, frame_buffer, str(snap_dir.parent), camera_id)

    # -- Write track summaries -------------------------------------------------
    summaries = [t.to_summary_dict() for t in tracker.all_tracks()]
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    logger.info("Track summary written: %s (%d tracks)", summary_path, len(summaries))

    # -- Final performance metrics ---------------------------------------------
    elapsed_total = get_monotonic_time() - processing_start
    proc_fps = frames_received / elapsed_total if elapsed_total > 0 else 0.0
    avg_inf_ms = total_inference_ms / max(frames_detected, 1)

    print("\n" + "=" * 65)
    print("  STEP 4 -- VEHICLE DETECTION PERFORMANCE REPORT")
    print("=" * 65)
    print(f"  Frames received          : {frames_received}")
    print(f"  Frames processed (det.)  : {frames_detected}")
    print(f"  Vehicles detected total  : {total_vehicles}")
    print(f"  Active tracks remaining  : {tracker.active_count()}")
    print(f"  Completed tracks         : {tracker.completed_count()}")
    print(f"  Stream reconnects        : {reader.health.reconnect_count}")
    print(f"  PTS discontinuities      : {reader.health.pts_discontinuities}")
    print(f"  Dropped frames           : {dropped_frames}")
    print(f"  Avg inference latency    : {avg_inf_ms:.1f} ms")
    print(f"  Processing throughput    : {proc_fps:.2f} fps (actual wall-clock)")
    print(f"  Elapsed time             : {elapsed_total:.1f} s")
    print(f"  JSONL output             : {jsonl_path}")
    print(f"  Track summary            : {summary_path}")
    print(f"  Snapshots dir            : {snap_dir}")
    print("=" * 65 + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Step 4 -- Sentinel CCTV Vehicle Detection & Tracking"
    )
    ap.add_argument("--camera-id", default=os.getenv("DEFAULT_CAMERA_ID", "cam01"),
                    help="Camera ID from normalized catalogue (default: cam01)")
    ap.add_argument("--catalogue",
                    default="data/catalogue/normalized/cameras.json")
    ap.add_argument("--stream-url", "--input", default=None,
                    help="Custom RTSP URL or path to local video file (overrides catalogue rtsp_url)")
    ap.add_argument("--frames", type=int, default=None,
                    help="Max frames to process (omit for continuous)")
    ap.add_argument("--confidence", type=float, default=0.40,
                    help="Minimum detection confidence 0-1 (default: 0.40)")
    ap.add_argument("--detector-interval", type=int, default=3,
                    help="Run detector every N frames (default: 3)")
    ap.add_argument("--best-frames", type=int, default=5,
                    help="Best candidate frames to keep per track (default: 5)")
    ap.add_argument("--no-display", action="store_true",
                    help="Run in headless mode (no OpenCV window)")
    ap.add_argument("--model", default="yolov8n.pt",
                    help="YOLO model name or path (default: yolov8n.pt)")
    ap.add_argument("--detections-dir", default="data/detections",
                    help="Output directory for detections (default: data/detections)")
    ap.add_argument("--snapshots-dir", default="data/snapshots",
                    help="Output directory for snapshots (default: data/snapshots)")

    args = ap.parse_args()

    run_detection(
        camera_id=args.camera_id,
        catalogue_path=args.catalogue,
        stream_url=args.stream_url,
        max_frames=args.frames,
        confidence=args.confidence,
        detector_interval=args.detector_interval,
        best_frames=args.best_frames,
        show_display=not args.no_display,
        model_name=args.model,
        detections_dir=args.detections_dir,
        snapshots_dir=args.snapshots_dir,
    )


if __name__ == "__main__":
    main()
