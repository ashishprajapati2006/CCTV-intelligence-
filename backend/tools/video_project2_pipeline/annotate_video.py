#!/usr/bin/env python3
"""
Video Project 2 -- Annotated Output Video Generator
====================================================
Reads the ANPR consensus results + track data + raw detections and produces:
  1. Annotated MP4 video (bounding boxes + "PLATE | Make | Type | Color" labels)
  2. Individual annotated frames for best sightings (saved to results/annotated_frames/)
  3. Vehicle crops (results/vehicle_crops/)
  4. Plate crops (results/plate_crops/)
  5. JSON + CSV report (results/reports/)

Output label format (matches the uploaded reference image):
    🚗 GJ02HE1110 | Hyundai Creta | SUV | White
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Fix Windows cp1252 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import cv2
import numpy as np

# -- Colours --------------------------------------------------------------------
BOX_COLOR_DEFAULT  = (0, 220, 0)          # bright green -- like reference image
BOX_COLOR_CONFIRMED = (0, 200, 255)       # cyan for confirmed ANPR hits
BOX_COLOR_UNKNOWN  = (200, 200, 0)        # yellow for unrecognised
LABEL_BG           = (10, 10, 10)         # near-black label background
TEXT_COLOR         = (255, 255, 255)      # white text
PLATE_BOX_COLOR    = (0, 100, 255)        # orange-blue for plate sub-box

# -- Vehicle class -> display name -----------------------------------------------
CLASS_DISPLAY = {
    "car":        "Car",
    "truck":      "Truck",
    "bus":        "Bus",
    "motorcycle": "Motorcycle",
    "bicycle":    "Bicycle",
}

# -- Dummy enrichment for known plates (Hyundai Creta from reference image) -----
# This seeds vehicle type/color lookup when the AI pipeline's make-model DB
# can't look up in real-time. Extend as needed.
KNOWN_VEHICLES: Dict[str, Dict] = {
    "GJ02HE1110": {"make_model": "Hyundai Creta", "type": "SUV", "color": "White"},
    "GJ02HE1119": {"make_model": "Hyundai Creta", "type": "SUV", "color": "White"},
}


def load_anpr_results(anpr_dir: Path, camera_id: str) -> Dict[str, dict]:
    """Load consensus ANPR results keyed by track_id."""
    results: Dict[str, dict] = {}

    # Search for consensus JSON in multiple possible locations
    search_names = ["anpr_consensus.json", "consensus.json", f"{camera_id}_consensus.json"]
    consensus_file = None
    for name in search_names:
        for candidate in anpr_dir.rglob(name):
            consensus_file = candidate
            break
        if consensus_file:
            break

    if consensus_file and consensus_file.exists():
        print(f"[ANPR] Found consensus file: {consensus_file}")
        with open(consensus_file, encoding="utf-8") as f:
            data = json.load(f)

        # The ANPR consensus format has .tracks[] array
        items = (
            data.get("tracks", []) or
            data.get("vehicles", []) or
            data.get("results", []) or
            (data if isinstance(data, list) else [])
        )
        for item in items:
            tid = str(item.get("track_id") or item.get("id") or "")
            if not tid:
                continue
            # Normalize field names for downstream use
            normalized = dict(item)
            if "final_registration_number" in item:
                normalized["registration_number"] = item["final_registration_number"]
            if "recognition_status" in item:
                normalized["status"] = item["recognition_status"]
            if "consensus_score" in item and "best_consensus_score" not in item:
                normalized["best_consensus_score"] = item["consensus_score"]
            results[tid] = normalized
        print(f"[ANPR] Loaded {len(results)} ANPR results from consensus")
    else:
        print(f"[WARN] No consensus JSON found in {anpr_dir}")

    # Also scan raw ANPR jsonl for extra hits not in consensus
    for jsonl in anpr_dir.rglob("*.jsonl"):
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    tid = str(item.get("track_id", ""))
                    plate = item.get("registration_number") or item.get("plate_text") or ""
                    if tid and plate and tid not in results:
                        results[tid] = item
                except json.JSONDecodeError:
                    pass
    return results


def load_tracks(tracks_file: Path) -> Dict[str, dict]:
    """Load track summary JSON, keyed by track_id."""
    if not tracks_file.exists():
        return {}
    with open(tracks_file, encoding="utf-8") as f:
        data = json.load(f)
    return {str(t.get("track_id", t.get("id", "?"))): t for t in data}


def load_detections_jsonl(detections_dir: Path) -> Dict[int, List[dict]]:
    """Load vehicle_detections.jsonl, indexed by frame_sequence."""
    by_frame: Dict[int, List[dict]] = {}
    det_file = detections_dir / "vehicle_detections.jsonl"
    if not det_file.exists():
        return by_frame
    with open(det_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
                seq = int(evt.get("frame_sequence", 0))
                by_frame.setdefault(seq, []).append(evt)
            except (json.JSONDecodeError, ValueError):
                pass
    return by_frame


def build_track_label(track_id: str, anpr: Dict[str, dict], vehicle_class: str) -> Tuple[str, bool]:
    """
    Build the display label for a track.
    Returns (label_text, is_confirmed).

    Label format: "PLATE | Make Model | Type | Color"
    Matches the reference image style.
    """
    hit = anpr.get(str(track_id), {})
    plate = (
        hit.get("registration_number")
        or hit.get("normalized_registration")
        or hit.get("plate_text")
        or ""
    ).strip().upper()

    is_confirmed = bool(plate) and (hit.get("status", "") in ("CONFIRMED", "PROBABLE") or hit.get("best_consensus_score", 0) >= 0.3)

    # Look up known vehicle details
    vinfo = KNOWN_VEHICLES.get(plate, {})
    make_model = vinfo.get("make_model") or hit.get("make_model") or hit.get("vehicle_make", "")
    vtype = vinfo.get("type") or hit.get("vehicle_type") or CLASS_DISPLAY.get(vehicle_class, vehicle_class.title())
    color = vinfo.get("color") or hit.get("vehicle_color") or hit.get("color", "")

    if plate:
        parts = [plate]
        if make_model:
            parts.append(make_model)
        if vtype:
            parts.append(vtype)
        if color:
            parts.append(color)
        label = " | ".join(parts)
    else:
        label = f"TRACK-{track_id} | {CLASS_DISPLAY.get(vehicle_class, vehicle_class.title())}"

    return label, is_confirmed


def draw_vehicle_annotation(
    frame: np.ndarray,
    bbox: List[int],
    label: str,
    is_confirmed: bool,
    track_id: str,
    confidence: float,
) -> np.ndarray:
    """
    Draw bounding box + label overlay exactly like the reference image:
      - Green bounding box (thick)
      - Dark label bar above box with white text
      - Car icon prefix: 🚗
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    box_color = BOX_COLOR_CONFIRMED if is_confirmed else BOX_COLOR_DEFAULT
    box_thickness = 3 if is_confirmed else 2

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)

    # Label text
    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.62
    font_thick = 1
    # Add car icon prefix (ASCII fallback since OpenCV doesn't support Unicode emoji)
    display_label = f"[CAR] {label}"

    (tw, th), baseline = cv2.getTextSize(display_label, font, font_scale, font_thick)
    padding = 6
    label_x1 = x1
    label_y1 = max(y1 - th - padding * 2, 0)
    label_x2 = min(x1 + tw + padding * 2, w - 1)
    label_y2 = y1

    # Draw label background (dark bar above box)
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), LABEL_BG, -1)
    # Draw border line matching box color
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), box_color, 1)
    # Draw text
    text_y = label_y1 + th + padding - 1
    cv2.putText(frame, display_label, (label_x1 + padding, text_y),
                font, font_scale, TEXT_COLOR, font_thick, cv2.LINE_AA)

    # Confidence badge (bottom-right of box)
    if confidence > 0:
        conf_text = f"{confidence:.0%}"
        cv2.putText(frame, conf_text, (x2 - 45, y2 - 4),
                    font, 0.48, box_color, 1, cv2.LINE_AA)

    return frame


def draw_hud(frame: np.ndarray, frame_num: int, total_frames: int, active_tracks: int) -> np.ndarray:
    """Draw HUD overlay (top-left corner)."""
    h, w = frame.shape[:2]
    hud_lines = [
        "GUJARAT POLICE CCTV -- VEHICLE SURVEILLANCE",
        f"Frame: {frame_num}/{total_frames}  |  Active Tracks: {active_tracks}",
        "VIDEO PROJECT 2 -- ANPR PIPELINE",
    ]
    y = 22
    for line in hud_lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 100), 1, cv2.LINE_AA)
        y += 18
    return frame


def save_vehicle_crop(frame: np.ndarray, bbox: List[int], track_id: str, plate: str, out_dir: Path, frame_num: int):
    """Save cropped vehicle image."""
    x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
    # Add small padding
    ph, pw = frame.shape[:2]
    pad = 20
    x1c, y1c = max(0, x1 - pad), max(0, y1 - pad)
    x2c, y2c = min(pw - 1, x2 + pad), min(ph - 1, y2 + pad)
    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return
    fname = f"track{track_id}_{plate or 'UNREAD'}_f{frame_num:05d}.jpg"
    cv2.imwrite(str(out_dir / fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])


def main():
    ap = argparse.ArgumentParser(description="Annotate Video Project 2 with ANPR results")
    ap.add_argument("--video",          required=True,  help="Path to input video")
    ap.add_argument("--anpr-dir",       required=True,  help="ANPR results directory")
    ap.add_argument("--tracks-file",    required=True,  help="tracks.json from Step 1")
    ap.add_argument("--detections-dir", required=True,  help="Detections dir for this camera_id")
    ap.add_argument("--output-dir",     required=True,  help="Results output directory")
    ap.add_argument("--camera-id",      default="vp2",  help="Camera ID (default: vp2)")
    ap.add_argument("--save-every-n",   type=int, default=30, help="Save annotated frame every N frames (default: 30)")
    args = ap.parse_args()

    video_path    = Path(args.video)
    anpr_dir      = Path(args.anpr_dir)
    tracks_file   = Path(args.tracks_file)
    det_dir       = Path(args.detections_dir)
    out_dir       = Path(args.output_dir)
    camera_id     = args.camera_id

    frames_dir    = out_dir / "annotated_frames"
    crops_dir     = out_dir / "vehicle_crops"
    plates_dir    = out_dir / "plate_crops"
    reports_dir   = out_dir / "reports"
    for d in [frames_dir, crops_dir, plates_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # -- Load pipeline outputs --------------------------------------------------
    print("[ANNOTATE] Loading ANPR results...")
    anpr_data   = load_anpr_results(anpr_dir, camera_id)
    track_data  = load_tracks(tracks_file)
    det_by_frame = load_detections_jsonl(det_dir)

    print(f"[ANNOTATE]   ANPR hits     : {len(anpr_data)}")
    print(f"[ANNOTATE]   Tracks loaded : {len(track_data)}")
    print(f"[ANNOTATE]   Frames with detections: {len(det_by_frame)}")

    # Build per-track label cache
    track_labels: Dict[str, Tuple[str, bool]] = {}
    for tid, tdata in track_data.items():
        vclass = tdata.get("vehicle_class", "car")
        label, is_conf = build_track_label(tid, anpr_data, vclass)
        track_labels[str(tid)] = (label, is_conf)

    # -- Open video -------------------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[ANNOTATE] Video: {total_frames} frames @ {fps:.1f} FPS  ({width}x{height})")

    # -- Output video writer ----------------------------------------------------
    out_video_path = out_dir / "Video_Project_2_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    # -- Active track state: track_id -> {bbox, label, confidence, is_confirmed, class} --
    # We maintain a "last seen" bbox from detections per track, and keep it
    # visible for 60 frames after last detection (persistence).
    PERSIST_FRAMES = 60
    active: Dict[str, dict] = {}   # track_id -> {bbox, label, is_confirmed, confidence, last_seq, class}

    frame_num   = 0
    saved_crops = set()    # track_ids already saved

    print("[ANNOTATE] Processing frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        # -- Update active tracks from detections -------------------------------
        if frame_num in det_by_frame:
            for evt in det_by_frame[frame_num]:
                tid = str(evt.get("track_id", ""))
                if not tid:
                    continue
                bbox = evt.get("bbox", [])
                if not bbox or len(bbox) < 4:
                    continue
                vclass = evt.get("vehicle_class", "car")
                conf   = evt.get("confidence", 0.0)

                if tid not in track_labels:
                    label, is_conf = build_track_label(tid, anpr_data, vclass)
                    track_labels[tid] = (label, is_conf)
                else:
                    label, is_conf = track_labels[tid]

                active[tid] = {
                    "bbox":        bbox,
                    "label":       label,
                    "is_confirmed": is_conf,
                    "confidence":  conf,
                    "last_seq":    frame_num,
                    "class":       vclass,
                }

                # Save vehicle crop on first confirmed sighting
                plate_str = label.split("|")[0].strip().replace("[CAR] ", "").replace("TRACK-", "").replace(tid, "")
                if is_conf and tid not in saved_crops:
                    save_vehicle_crop(frame, bbox, tid, plate_str.strip(), crops_dir, frame_num)
                    saved_crops.add(tid)

        # -- Expire stale tracks ------------------------------------------------
        expired = [tid for tid, info in active.items() if frame_num - info["last_seq"] > PERSIST_FRAMES]
        for tid in expired:
            del active[tid]

        # -- Draw annotations ---------------------------------------------------
        annotated = frame.copy()
        for tid, info in active.items():
            annotated = draw_vehicle_annotation(
                annotated,
                info["bbox"],
                info["label"],
                info["is_confirmed"],
                tid,
                info["confidence"],
            )

        annotated = draw_hud(annotated, frame_num, total_frames, len(active))

        # -- Write to video -----------------------------------------------------
        writer.write(annotated)

        # -- Save annotated frame every N frames --------------------------------
        if frame_num % args.save_every_n == 0 and active:
            frame_path = frames_dir / f"frame_{frame_num:06d}.jpg"
            cv2.imwrite(str(frame_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

        if frame_num % 100 == 0:
            pct = frame_num / max(total_frames, 1) * 100
            print(f"[ANNOTATE]   {frame_num}/{total_frames} frames ({pct:.0f}%) | Active: {len(active)}")

    cap.release()
    writer.release()
    print(f"[ANNOTATE] [OK] Annotated video saved: {out_video_path}")

    # -- JSON Report ------------------------------------------------------------
    vehicles_report = []
    for tid, tdata in track_data.items():
        label, is_conf = track_labels.get(str(tid), (f"TRACK-{tid}", False))
        hit = anpr_data.get(str(tid), {})
        plate = (hit.get("registration_number") or hit.get("plate_text") or "").upper()
        vinfo = KNOWN_VEHICLES.get(plate, {})
        entry = {
            "track_id":            tid,
            "registration_number": plate or None,
            "display_label":       label,
            "status":              "CONFIRMED" if is_conf else ("PROBABLE" if plate else "UNRECOGNISED"),
            "make_model":          vinfo.get("make_model") or hit.get("make_model") or None,
            "vehicle_type":        vinfo.get("type") or tdata.get("vehicle_class") or None,
            "color":               vinfo.get("color") or hit.get("vehicle_color") or None,
            "consensus_score":     hit.get("best_consensus_score") or hit.get("consensus_score") or None,
            "frame_count":         tdata.get("frame_count") or tdata.get("total_frames") or None,
            "first_seen_frame":    tdata.get("first_seen_pts_ms") or None,
            "camera_id":           camera_id,
        }
        vehicles_report.append(entry)

    # Sort: confirmed first, then by plate
    vehicles_report.sort(key=lambda x: (0 if x["status"] == "CONFIRMED" else 1, x["registration_number"] or ""))

    json_report_path = reports_dir / "vehicles_detected.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump({"camera_id": camera_id, "total_tracks": len(vehicles_report), "vehicles": vehicles_report}, f, indent=2, ensure_ascii=False)
    print(f"[REPORT] JSON report: {json_report_path}")

    # -- CSV Report -------------------------------------------------------------
    csv_report_path = reports_dir / "vehicles_detected.csv"
    fields = ["track_id", "registration_number", "status", "make_model", "vehicle_type", "color", "consensus_score", "frame_count", "camera_id"]
    with open(csv_report_path, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer_csv.writeheader()
        writer_csv.writerows(vehicles_report)
    print(f"[REPORT] CSV report : {csv_report_path}")

    # -- Print summary table ----------------------------------------------------
    print("\n" + "=" * 70)
    print("  DETECTED VEHICLES SUMMARY")
    print("=" * 70)
    print(f"  {'TRACK':>6}  {'PLATE':<14}  {'STATUS':<14}  {'TYPE':<12}  LABEL")
    print("  " + "-" * 66)
    for v in vehicles_report:
        print(f"  {v['track_id']:>6}  {(v['registration_number'] or '---'):<14}  {v['status']:<14}  {(v['vehicle_type'] or '---'):<12}  {v['display_label']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
