#!/usr/bin/env python3
"""
Video Project 2 -- Dedicated Vehicle Detection & ANPR Pipeline
=============================================================
Processes "Video Project 2.mp4" through the complete AI pipeline:
  Step 1: Vehicle detection (YOLOv8, every frame, high confidence)
  Step 2: Plate detection & cropping
  Step 3: ANPR / OCR with multi-frame consensus (Indian plate format)
  Step 4: Annotated output video with bounding boxes + labels
  Step 5: JSON & CSV reports

Target: Maximum accuracy (100%) on the few vehicles in the video.

Run from project root:
    python video_project2_pipeline/run_pipeline.py
"""
import subprocess
import sys
import json
import time
from pathlib import Path

# -- Paths ----------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parent.parent          # CCTV Platform/
PIPELINE    = Path(__file__).resolve().parent                 # video_project2_pipeline/
VIDEO_SRC   = ROOT / "Video Project 2.mp4"
PROC_DIR    = PIPELINE / "processing"
RESULTS_DIR = PIPELINE / "results"
CATALOGUE   = PIPELINE / "processing" / "camera_catalogue.json"

CAMERA_ID   = "vp2"

# -- Verify video exists --------------------------------------------------------
if not VIDEO_SRC.exists():
    print(f"[ERROR] Video not found: {VIDEO_SRC}")
    sys.exit(1)

VIDEO_MB = VIDEO_SRC.stat().st_size / (1024 * 1024)
print("=" * 70)
print("  GUJARAT POLICE CCTV -- VIDEO PROJECT 2 DEDICATED PIPELINE")
print("=" * 70)
print(f"  Video   : {VIDEO_SRC.name}  ({VIDEO_MB:.1f} MB)")
print(f"  Camera  : {CAMERA_ID}")
print(f"  Results : {RESULTS_DIR}")
print("=" * 70)

# -- Write single-video catalogue -----------------------------------------------
catalogue = [{
    "camera_id": CAMERA_ID,
    "name": "Video Project 2",
    "rtsp_url": str(VIDEO_SRC),
    "codec": "H264",
    "location": "Gujarat Road Segment",
    "department": "Gujarat Traffic Police",
    "status": "online",
    "is_spatial": False,
}]
CATALOGUE.parent.mkdir(parents=True, exist_ok=True)
with open(CATALOGUE, "w", encoding="utf-8") as f:
    json.dump(catalogue, f, indent=2)
print(f"[INFO] Camera catalogue written: {CATALOGUE}")

# -- Utility --------------------------------------------------------------------
def run(cmd: list, desc: str) -> bool:
    print(f"\n{'--'*35}")
    print(f"  >>  {desc}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}")
    print(f"{'--'*35}")
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
        print(f"  [OK]  DONE in {time.time()-t0:.1f}s\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [FAIL]  FAILED (code {e.returncode}) after {time.time()-t0:.1f}s\n")
        return False

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PY = sys.executable

# -- STEP 1: Vehicle Detection (every frame, high accuracy) --------------------
print("\n[STEP 1] Vehicle Detection (YOLOv8, confidence>=0.35, interval=1)")
ok = run([
    PY, str(ROOT / "scripts" / "run_vehicle_detection.py"),
    "--camera-id",       CAMERA_ID,
    "--catalogue",       str(CATALOGUE),
    "--detector-interval", "1",          # detect EVERY frame
    "--confidence",      "0.35",         # slightly lower threshold to catch all vehicles
    "--best-frames",     "10",           # keep top 10 frames per track
    "--detections-dir",  str(PROC_DIR / "detections"),
    "--snapshots-dir",   str(PROC_DIR / "snapshots"),
    "--no-display",
], "Step 1 -- Vehicle Detection & Tracking")

if not ok:
    print("[WARN] Vehicle detection failed or incomplete. Continuing anyway...")

# -- STEP 2: Plate Detection ----------------------------------------------------
print("\n[STEP 2] Plate Detection (low threshold to catch all candidates)")
tracks_file = PROC_DIR / "detections" / CAMERA_ID / "tracks.json"
ok2 = run([
    PY, str(ROOT / "scripts" / "run_plate_detection.py"),
    "--camera-id",       CAMERA_ID,
    "--tracks",          str(tracks_file),
    "--catalogue",       str(CATALOGUE),
    "--detections",      str(PROC_DIR / "detections"),
    "--snapshots",       str(PROC_DIR / "snapshots"),
    "--plate-confidence", "0.10",        # low threshold -> more candidates
    "--no-display",
], "Step 2 -- License Plate Detection & Crop")

# -- STEP 3: ANPR / OCR + Consensus --------------------------------------------
print("\n[STEP 3] ANPR / OCR with Indian plate format & multi-frame consensus")
plate_candidates = PROC_DIR / "detections" / CAMERA_ID / "plate_candidates.jsonl"
ok3 = run([
    PY, str(ROOT / "scripts" / "run_anpr.py"),
    "--camera-id",          CAMERA_ID,
    "--input",              str(plate_candidates),
    "--output",             str(PROC_DIR / "anpr"),
    "--min-ocr-confidence", "0.05",      # low floor -- keep everything
    "--min-consensus",      "0.10",
    "--min-probable",       "0.05",
    "--plate-format",       "indian",    # GJ-format plates
    "--cpu",                             # CPU mode (safe for all machines)
    "--no-display",
], "Step 3 -- ANPR / OCR + Indian Plate Consensus")

# -- STEP 4: Annotated Output Video --------------------------------------------
print("\n[STEP 4] Generating annotated output video with bounding boxes + labels")
ok4 = run([
    PY, str(PIPELINE / "annotate_video.py"),
    "--video",       str(VIDEO_SRC),
    "--anpr-dir",    str(PROC_DIR / "anpr"),
    "--tracks-file", str(tracks_file),
    "--detections-dir", str(PROC_DIR / "detections" / CAMERA_ID),
    "--output-dir",  str(RESULTS_DIR),
    "--camera-id",   CAMERA_ID,
], "Step 4 -- Annotated Output Video + Frame Exports + Reports")

# -- Summary --------------------------------------------------------------------
print("\n" + "=" * 70)
print("  VIDEO PROJECT 2 PIPELINE COMPLETE")
print("=" * 70)
print(f"  Annotated video  : {RESULTS_DIR / 'Video_Project_2_annotated.mp4'}")
print(f"  Vehicle report   : {RESULTS_DIR / 'reports' / 'vehicles_detected.json'}")
print(f"  CSV summary      : {RESULTS_DIR / 'reports' / 'vehicles_detected.csv'}")
print(f"  Annotated frames : {RESULTS_DIR / 'annotated_frames'}/")
print(f"  Vehicle crops    : {RESULTS_DIR / 'vehicle_crops'}/")
print(f"  Plate crops      : {RESULTS_DIR / 'plate_crops'}/")
print("=" * 70 + "\n")
