#!/usr/bin/env python3
"""Step 5 - Best-Frame Selection + License Plate Detection.

Executes Step 5 pipeline:
  1. Rank vehicle candidate frames for each track using composite quality scoring.
  2. Select top-N best vehicle frames per track.
  3. Run license plate region detection on selected vehicle frames (ROI crops).
  4. Validate plate bounding boxes & score plate crop quality.
  5. Generate OCR preprocessing variants (original, grayscale, upscaled, contrast, sharpened).
  6. Deduplicate multi-frame plate candidates per track and export to JSONL and JSON summary.

MODES:
  Offline Mode (preferred for fast testing without live stream):
    python scripts/run_plate_detection.py --camera-id cam01 --tracks data/detections/cam01/tracks.json

  Live Stream Mode:
    python scripts/run_plate_detection.py --camera-id cam01 --frames 500 --no-display

TIMING PRINCIPLES:
  - media_pts_ms = cap.get(CAP_PROP_POS_MSEC) (authoritative video container timeline)
  - local_receive_monotonic = time.monotonic() (pipeline latency diagnostics only)
  - CAP_PROP_FPS is NEVER used for timing.

SCOPE BOUNDARY - THIS SCRIPT DOES NOT:
  - Perform ANPR or OCR text recognition
  - Assign registration numbers (e.g. GJ01AB1234)
  - Match against a watchlist
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src.ai.plate_detector import PlateDetector
from src.ai.plate_quality import validate_plate_bbox, score_plate
from src.ai.plate_preprocessor import crop_plate_with_margin, save_plate_crops
from src.ai.candidate_manager import CandidateManager
from src.ai.schemas import PlateCandidate, VehicleFrameCandidate
from src.ai.frame_selector import (
    evaluate_frame_extended,
    select_top_vehicle_frames,
    FrameQualityWeights,
)
from src.catalogue.models import Camera
from src.common.logging import get_logger
from src.common.time import get_monotonic_time

logger = get_logger("plate_detection")


def _load_camera(camera_id: str, catalogue_path: str) -> Camera:
    with open(catalogue_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        cam = Camera.from_dict(item)
        if cam.camera_id == camera_id:
            return cam
    available = [i.get("camera_id", i.get("id", "?")) for i in data]
    raise ValueError(f"Camera '{camera_id}' not in catalogue. Available: {available}")


def process_vehicle_track_frames(
    camera_id: str,
    track_id: str,
    vehicle_class: str,
    frame_images: List[Tuple[int, Optional[float], float, np.ndarray, List[int], float]],
    # List of (frame_id, pts_ms, receive_mono, frame_img, vehicle_bbox, det_conf)
    plate_detector: PlateDetector,
    candidate_mgr: CandidateManager,
    best_vehicle_frames: int = 5,
    plate_confidence_thresh: float = 0.10,
    save_crops: bool = True,
    snapshot_dir: str = "data/snapshots",
    show_display: bool = False,
) -> Tuple[int, int, int, float, float, float]:
    """Process a single vehicle track through Step 5 pipeline.

    Returns:
        (num_selected_v_frames, raw_plate_dets, valid_plates,
         frame_select_ms, plate_det_ms, preproc_ms)
    """
    t_start = get_monotonic_time()

    # 1. Score all vehicle frames in this track
    v_candidates: List[VehicleFrameCandidate] = []
    for frame_id, pts_ms, mono_time, frame_img, bbox, det_conf in frame_images:
        cand = evaluate_frame_extended(
            camera_id=camera_id,
            track_id=track_id,
            vehicle_class=vehicle_class,
            detection_confidence=det_conf,
            frame=frame_img,
            bbox=bbox,
            media_pts_ms=pts_ms,
            local_receive_monotonic=mono_time,
            frame_sequence=frame_id,
        )
        v_candidates.append(cand)

    # 2. Select top-N vehicle frames
    top_v_candidates = select_top_vehicle_frames(v_candidates, top_n=best_vehicle_frames)
    t_select_done = get_monotonic_time()
    frame_select_ms = (t_select_done - t_start) * 1000.0

    # Build lookup map for top frames
    top_frame_ids = {c.frame_id: c for c in top_v_candidates}

    raw_plate_dets = 0
    valid_plates = 0
    plate_det_ms = 0.0
    preproc_ms = 0.0

    plate_index = 1

    for frame_id, pts_ms, mono_time, frame_img, vehicle_bbox, det_conf in frame_images:
        if frame_id not in top_frame_ids:
            continue

        v_cand = top_frame_ids[frame_id]
        h, w = frame_img.shape[:2]

        # Extract vehicle crop ROI
        vx1, vy1, vx2, vy2 = vehicle_bbox
        vx1c, vy1c = max(0, vx1), max(0, vy1)
        vx2c, vy2c = min(w, vx2), min(h, vy2)
        vehicle_crop = frame_img[vy1c:vy2c, vx1c:vx2c]

        if vehicle_crop.size == 0:
            continue

        # 3. Plate detection on vehicle ROI crop
        t_det_start = get_monotonic_time()
        detections, det_latency = plate_detector.detect(
            crop=vehicle_crop,
            offset_x=vx1c,
            offset_y=vy1c,
        )
        plate_det_ms += det_latency
        raw_plate_dets += len(detections)

        for det in detections:
            full_plate_bbox = det.bbox  # already converted to full-frame coords

            # 4. Bounding box validation
            is_valid, rejection_reason = validate_plate_bbox(
                plate_bbox=full_plate_bbox,
                frame_h=h,
                frame_w=w,
                plate_confidence=det.confidence,
                min_confidence=plate_confidence_thresh,
            )

            # Crop plate region from full frame for quality scoring & saving
            t_preproc_start = get_monotonic_time()
            plate_crop, clamped_bbox = crop_plate_with_margin(
                frame_img, full_plate_bbox, margin_percent=0.10
            )

            # 5. Plate quality scoring
            quality_metrics = score_plate(plate_crop)

            orig_crop_path = None
            processed_paths: Dict[str, str] = {}

            if is_valid and save_crops and plate_crop.size > 0:
                orig_crop_path, processed_paths = save_plate_crops(
                    crop=plate_crop,
                    camera_id=camera_id,
                    track_id=track_id,
                    frame_id=frame_id,
                    pts_ms=pts_ms,
                    plate_index=plate_index,
                    snapshot_dir=snapshot_dir,
                    generate_variants=True,
                )
                plate_index += 1

            preproc_ms += (get_monotonic_time() - t_preproc_start) * 1000.0

            pw = max(0, full_plate_bbox[2] - full_plate_bbox[0])
            ph = max(0, full_plate_bbox[3] - full_plate_bbox[1])

            plate_cand = PlateCandidate(
                camera_id=camera_id,
                track_id=track_id,
                frame_id=frame_id,
                pts_ms=pts_ms,
                has_valid_pts=(pts_ms is not None and pts_ms > 0),
                local_receive_monotonic=mono_time,
                plate_bbox=full_plate_bbox,
                plate_confidence=det.confidence,
                detector_version=det.detector_version,
                plate_quality_score=quality_metrics.composite_score,
                plate_width=pw,
                plate_height=ph,
                sharpness_score=quality_metrics.sharpness_score,
                brightness_score=quality_metrics.brightness_score,
                contrast_score=quality_metrics.contrast_score,
                original_crop_path=orig_crop_path,
                processed_crop_paths=processed_paths,
                rejection_reason=rejection_reason,
            )

            candidate_mgr.add_candidate(plate_cand)

            if is_valid:
                valid_plates += 1
            else:
                logger.debug(
                    "[%s/%s] Plate candidate rejected: %s (conf=%.2f)",
                    camera_id, track_id, rejection_reason, det.confidence
                )

            # Debug display if enabled
            if show_display and is_valid:
                disp = frame_img.copy()
                cv2.rectangle(disp, (vx1c, vy1c), (vx2c, vy2c), (0, 255, 0), 2)
                cv2.rectangle(disp, (full_plate_bbox[0], full_plate_bbox[1]),
                              (full_plate_bbox[2], full_plate_bbox[3]), (0, 0, 255), 2)
                lbl = f"{track_id} | Plate: {det.confidence:.2f} | Quality: {quality_metrics.composite_score:.2f}"
                cv2.putText(disp, lbl, (full_plate_bbox[0], max(0, full_plate_bbox[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                try:
                    cv2.imshow("Step 5 Debug", disp)
                    cv2.waitKey(1)
                except cv2.error:
                    pass

    return len(top_v_candidates), raw_plate_dets, valid_plates, frame_select_ms, plate_det_ms, preproc_ms


def run_offline_plate_detection(
    camera_id: str,
    tracks_path: str,
    snapshots_dir: str = "data/snapshots",
    detections_dir: str = "data/detections",
    best_vehicle_frames: int = 5,
    best_plate_candidates: int = 5,
    plate_confidence: float = 0.10,
    save_crops: bool = True,
    show_display: bool = False,
) -> None:
    """Run Step 5 plate detection using saved Step 4 track summaries and snapshot images."""
    print("=" * 65)
    print("  GUJARAT POLICE CCTV - PLATE DETECTION STEP 5 (OFFLINE MODE)")
    print("=" * 65)
    print(f"  Camera ID            : {camera_id}")
    print(f"  Track summary input  : {tracks_path}")
    print(f"  Snapshots input      : {snapshots_dir}")
    print(f"  Best vehicle frames  : {best_vehicle_frames}")
    print(f"  Best plate candidates: {best_plate_candidates}")
    print(f"  Plate confidence     : >= {plate_confidence}")
    print("=" * 65)
    print("  NOTE: No ANPR/OCR text recognition is performed in Step 5.")
    print("=" * 65 + "\n")

    if not Path(tracks_path).exists():
        logger.error("Track file not found: %s. Run Step 4 first.", tracks_path)
        sys.exit(1)

    with open(tracks_path, "r", encoding="utf-8") as f:
        tracks_data = json.load(f)

    logger.info("Loaded %d vehicle track summaries from %s", len(tracks_data), tracks_path)

    detector = PlateDetector(confidence_threshold=plate_confidence)
    logger.info("Using plate detector version: %s", detector.detector_version)

    cand_mgr = CandidateManager(best_plates_per_track=best_plate_candidates)

    total_frames_examined = 0
    total_tracks_examined = len(tracks_data)
    total_v_candidates_retained = 0
    total_raw_plate_dets = 0
    total_valid_plates = 0

    t_total_start = get_monotonic_time()
    t_total_select_ms = 0.0
    t_total_det_ms = 0.0
    t_total_preproc_ms = 0.0

    # Load frame-specific vehicle bboxes from vehicle_detections.jsonl if available
    det_jsonl_path = Path(detections_dir) / camera_id / "vehicle_detections.jsonl"
    frame_bbox_map: Dict[Tuple[str, int], List[int]] = {}
    if det_jsonl_path.exists():
        logger.info("Loading frame detection bboxes from %s...", det_jsonl_path)
        try:
            with open(det_jsonl_path, "r", encoding="utf-8") as f_det:
                for line in f_det:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    t_id = rec.get("track_id")
                    f_seq = rec.get("frame_sequence")
                    b = rec.get("bbox")
                    if t_id and f_seq is not None and b:
                        frame_bbox_map[(t_id, f_seq)] = b
            logger.info("Loaded %d frame-level bboxes for precision vehicle cropping.", len(frame_bbox_map))
        except Exception as e:
            logger.warning("Could not index detection JSONL: %s", e)

    for track in tracks_data:
        track_id = track["track_id"]
        v_class = track.get("vehicle_class", "car")
        best_paths = track.get("best_frame_paths", [])
        if not best_paths:
            # Fallback: scan snapshot directory on disk for this track
            snap_cam_dir = Path(snapshots_dir) / camera_id
            if snap_cam_dir.exists():
                disk_matches = sorted(list(snap_cam_dir.glob(f"{camera_id}_{track_id}_*.jpg")))
                if disk_matches:
                    best_paths = [str(p) for p in disk_matches[:best_vehicle_frames]]

        if not best_paths:
            logger.debug("[%s] Track %s has no snapshot image paths.", camera_id, track_id)
            continue

        frame_images = []
        for path_str in best_paths:
            # Resolve relative/absolute path
            img_path = Path(path_str)
            if not img_path.exists():
                # try relative to project root
                img_path = Path.cwd() / path_str

            if not img_path.exists():
                logger.warning("[%s] Snapshot missing: %s", camera_id, path_str)
                continue

            frame_img = cv2.imread(str(img_path))
            if frame_img is None:
                continue

            total_frames_examined += 1

            # Parse frame metadata from filename pattern: cam01_TRK-0001_frame000036_pts2320.jpg
            fname = img_path.stem
            parts = fname.split("_")
            seq = 0
            pts_ms = None
            for p in parts:
                if p.startswith("frame"):
                    try: seq = int(p.replace("frame", ""))
                    except ValueError: pass
                elif p.startswith("pts"):
                    try:
                        pval = p.replace("pts", "")
                        if pval != "noPTS":
                            pts_ms = float(pval)
                    except ValueError: pass

            # Use exact detection bbox for this specific frame if available; fallback to last_bbox or full frame
            bbox = frame_bbox_map.get(
                (track_id, seq),
                track.get("last_bbox", [0, 0, frame_img.shape[1], frame_img.shape[0]])
            )
            det_conf = track.get("mean_confidence", 0.70)
            mono = get_monotonic_time()

            frame_images.append((seq, pts_ms, mono, frame_img, bbox, det_conf))

        if not frame_images:
            continue

        v_retained, raw_dets, valid_p, sel_ms, det_ms, prep_ms = process_vehicle_track_frames(
            camera_id=camera_id,
            track_id=track_id,
            vehicle_class=v_class,
            frame_images=frame_images,
            plate_detector=detector,
            candidate_mgr=cand_mgr,
            best_vehicle_frames=best_vehicle_frames,
            plate_confidence_thresh=plate_confidence,
            save_crops=save_crops,
            snapshot_dir=snapshots_dir,
            show_display=show_display,
        )

        total_v_candidates_retained += v_retained
        total_raw_plate_dets += raw_dets
        total_valid_plates += valid_p
        t_total_select_ms += sel_ms
        t_total_det_ms += det_ms
        t_total_preproc_ms += prep_ms

    if show_display:
        cv2.destroyAllWindows()

    # ── Export outputs ────────────────────────────────────────────────────────
    det_out_dir = Path(detections_dir) / camera_id
    det_out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = str(det_out_dir / "plate_candidates.jsonl")
    summary_path = str(det_out_dir / "plate_summary.json")

    retained_count = cand_mgr.export_jsonl(jsonl_path)
    summary_data = cand_mgr.export_summary_json(summary_path, camera_id)

    total_elapsed = get_monotonic_time() - t_total_start

    print("\n" + "=" * 65)
    print("  STEP 5 -- LICENSE PLATE DETECTION PERFORMANCE REPORT")
    print("=" * 65)
    print(f"  Detector engine          : {detector.detector_version}")
    print(f"  Frames examined          : {total_frames_examined}")
    print(f"  Vehicle tracks examined  : {total_tracks_examined}")
    print(f"  Vehicle candidates kept  : {total_v_candidates_retained}")
    print(f"  Raw plate detections     : {total_raw_plate_dets}")
    print(f"  Valid plate candidates   : {total_valid_plates}")
    print(f"  Deduplicated final crops : {retained_count}")
    print(f"  Rejected detections      : {summary_data['total_raw_rejected']}")
    print(f"  Frame selection latency  : {t_total_select_ms:.1f} ms total")
    print(f"  Plate detection latency  : {t_total_det_ms:.1f} ms total")
    print(f"  Preprocessing latency    : {t_total_preproc_ms:.1f} ms total")
    print(f"  Total processing time    : {total_elapsed:.2f} s")
    print(f"  JSONL output             : {jsonl_path}")
    print(f"  Summary JSON             : {summary_path}")
    print(f"  Plates directory         : data/snapshots/{camera_id}/plates/")
    print(f"  Processed directory      : data/snapshots/{camera_id}/plates_processed/")
    print("=" * 65 + "\n")


def run_live_plate_detection(
    camera_id: str,
    catalogue_path: str = "data/catalogue/normalized/cameras.json",
    max_frames: Optional[int] = 500,
    best_vehicle_frames: int = 5,
    best_plate_candidates: int = 5,
    plate_confidence: float = 0.35,
    save_crops: bool = True,
    show_display: bool = False,
) -> None:
    """Run Step 4 vehicle detection + Step 5 plate detection on live RTSP stream."""
    from scripts.run_vehicle_detection import run_detection as run_step4

    logger.info("Running Step 4 vehicle detection on live camera '%s' first...", camera_id)
    run_step4(
        camera_id=camera_id,
        catalogue_path=catalogue_path,
        max_frames=max_frames,
        show_display=show_display,
    )

    tracks_file = f"data/detections/{camera_id}/tracks.json"
    run_offline_plate_detection(
        camera_id=camera_id,
        tracks_path=tracks_file,
        best_vehicle_frames=best_vehicle_frames,
        best_plate_candidates=best_plate_candidates,
        plate_confidence=plate_confidence,
        save_crops=save_crops,
        show_display=show_display,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Step 5 - Sentinel CCTV Best-Frame Selection & License Plate Detection"
    )
    ap.add_argument("--camera-id", default=os.getenv("DEFAULT_CAMERA_ID", "cam01"),
                    help="Camera ID from normalized catalogue (default: cam01)")
    ap.add_argument("--tracks", default=None,
                    help="Path to saved Step 4 tracks.json for offline processing")
    ap.add_argument("--snapshots", default="data/snapshots",
                    help="Base snapshot directory (default: data/snapshots)")
    ap.add_argument("--detections", default="data/detections",
                    help="Base detections directory (default: data/detections)")
    ap.add_argument("--catalogue", default="data/catalogue/normalized/cameras.json")
    ap.add_argument("--frames", type=int, default=None,
                    help="Max frames to process for live mode (default: continuous)")
    ap.add_argument("--best-frames", type=int, default=5,
                    help="Top-N vehicle frames to keep per track (default: 5)")
    ap.add_argument("--best-plates", type=int, default=5,
                    help="Top-N plate candidates to keep per track (default: 5)")
    ap.add_argument("--plate-confidence", type=float, default=0.10,
                    help="Minimum plate detector confidence 0-1 (default: 0.10)")
    ap.add_argument("--no-display", action="store_true",
                    help="Run in headless mode (no debug window)")
    ap.add_argument("--save-crops", action="store_true", default=True,
                    help="Save plate crops and preprocessed variants (default: True)")
    ap.add_argument("--no-save-crops", action="store_false", dest="save_crops")

    args = ap.parse_args()

    # Determine execution mode: Offline if --tracks supplied or if saved tracks exist
    default_tracks = f"data/detections/{args.camera_id}/tracks.json"
    tracks_to_use = args.tracks or (default_tracks if Path(default_tracks).exists() and not args.frames else None)

    if tracks_to_use:
        run_offline_plate_detection(
            camera_id=args.camera_id,
            tracks_path=tracks_to_use,
            snapshots_dir=args.snapshots,
            detections_dir=args.detections,
            best_vehicle_frames=args.best_frames,
            best_plate_candidates=args.best_plates,
            plate_confidence=args.plate_confidence,
            save_crops=args.save_crops,
            show_display=not args.no_display,
        )
    else:
        run_live_plate_detection(
            camera_id=args.camera_id,
            catalogue_path=args.catalogue,
            max_frames=args.frames,
            best_vehicle_frames=args.best_frames,
            best_plate_candidates=args.best_plates,
            plate_confidence=args.plate_confidence,
            save_crops=args.save_crops,
            show_display=not args.no_display,
        )


if __name__ == "__main__":
    main()
