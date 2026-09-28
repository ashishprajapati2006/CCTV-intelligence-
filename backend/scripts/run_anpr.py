#!/usr/bin/env python3
"""Step 6 - ANPR / OCR + Multi-Frame Consensus CLI.

Executes Step 6 pipeline:
  1. Consumes Step 5 plate candidates (data/detections/<camera_id>/plate_candidates.jsonl).
  2. Evaluates candidate eligibility & logs rejection diagnostics.
  3. Executes OCR Preprocessing Cascade on eligible candidates.
  4. Normalizes text & logs character correction audits.
  5. Validates text against Indian vehicle registration format patterns.
  6. Computes composite evidence score for each observation.
  7. Executes multi-frame consensus across distinct video frames.
  8. Outputs raw observation logs, consensus JSON, and debug crops.

Usage examples:
  python scripts/run_anpr.py --camera-id cam01
  python scripts/run_anpr.py --camera-id cam01 --ocr-engine easyocr --debug
  python scripts/run_anpr.py --camera-id cam01 --min-ocr-confidence 0.20 --no-display
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src.ai.anpr import (
    ANPRRecognizer,
    MultiFrameConsensusEngine,
    OCREngineFacade,
    CandidateEligibilityEvaluator,
)
from src.common.logging import get_logger
from src.common.time import get_monotonic_time

logger = get_logger("run_anpr")


def run_anpr_pipeline(
    camera_id: str,
    input_jsonl_path: str,
    output_dir: str = "data/anpr",
    min_ocr_confidence: float = 0.02,
    min_consensus_score: float = 0.05,
    min_probable_score: float = 0.03,
    min_confirmed_frames: int = 1,
    max_candidates_per_track: int = 5,
    save_results: bool = True,
    save_debug_crops: bool = False,
    show_display: bool = False,
    engine_name: str = "easyocr",
    use_gpu: Optional[bool] = None,
    plate_format: str = "auto",
) -> None:
    print("=" * 65)
    print("  GUJARAT POLICE CCTV - ANPR / OCR & CONSENSUS (STEP 6)")
    print("=" * 65)
    print(f"  Camera ID            : {camera_id}")
    print(f"  Input candidates     : {input_jsonl_path}")
    print(f"  Output directory     : {output_dir}/{camera_id}")
    print(f"  OCR Engine           : {engine_name}")
    print(f"  Min OCR confidence   : >= {min_ocr_confidence}")
    print(f"  Min confirmed score  : >= {min_consensus_score} (frames: {min_confirmed_frames})")
    print(f"  Min probable score   : >= {min_probable_score}")
    print(f"  Plate format         : {plate_format.upper()}")
    print(f"  Max cands / track    : {max_candidates_per_track}")
    print(f"  Display              : {'YES' if show_display else 'HEADLESS'}")
    print("=" * 65)
    print("  NOTE: No watchlist matching or alert generation is performed in Step 6.")
    print("=" * 65 + "\n")

    input_file = Path(input_jsonl_path)
    if not input_file.exists():
        logger.error("Input candidate file not found: %s. Run Step 5 first.", input_jsonl_path)
        sys.exit(1)

    # 1. Load Step 5 candidates grouped by track_id
    candidates_by_track: Dict[str, List[Dict]] = defaultdict(list)
    total_candidates_read = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            item = json.loads(line_str)
            candidates_by_track[item["track_id"]].append(item)
            total_candidates_read += 1

    logger.info(
        "Loaded %d plate candidates across %d tracks from %s",
        total_candidates_read, len(candidates_by_track), input_jsonl_path
    )

    # 2. Initialize ANPR Engine
    consensus_engine = MultiFrameConsensusEngine(
        min_confirmed_score=min_consensus_score,
        min_probable_score=min_probable_score,
        min_confirmed_frames=min_confirmed_frames,
        enable_fuzzy_clustering=True,
        plate_format=plate_format,
    )

    eligibility_evaluator = CandidateEligibilityEvaluator(
        min_plate_width=8,
        min_plate_height=4,
        min_plate_area=32,
        min_detection_confidence=0.05,
        min_quality_score=0.05,
    )

    ocr_facade = OCREngineFacade(preferred_engine=engine_name, use_gpu=use_gpu)
    anpr = ANPRRecognizer(
        ocr_engine=ocr_facade,
        min_ocr_confidence=min_ocr_confidence,
        consensus_engine=consensus_engine,
        eligibility_evaluator=eligibility_evaluator,
        enable_cascade=True,
    )

    # 3. Processing loop
    all_raw_observations = []
    all_track_consensus = []
    all_eligibility_results = []

    counts = {
        "CONFIRMED": 0,
        "PROBABLE": 0,
        "UNCERTAIN": 0,
        "UNREADABLE": 0,
    }
    rejection_counts: Dict[str, int] = defaultdict(int)

    t_start = get_monotonic_time()

    for track_id in sorted(candidates_by_track.keys()):
        cands = candidates_by_track[track_id][:max_candidates_per_track]

        consensus, obs_list, elig_list = anpr.process_track_candidates(
            camera_id=camera_id,
            track_id=track_id,
            track_candidates=cands,
        )

        all_raw_observations.extend(obs_list)
        all_track_consensus.append(consensus)
        all_eligibility_results.extend(elig_list)
        counts[consensus.recognition_status] += 1

        for el in elig_list:
            if not el.is_eligible:
                rejection_counts[el.rejection_reason] += 1

        logger.info(
            "[%s/%s] Status: %s | Reg: %s | Score: %.2f | Support: %d frame(s)",
            camera_id, track_id, consensus.recognition_status,
            consensus.final_registration_number or "N/A",
            consensus.consensus_score, consensus.supporting_frame_count
        )

    t_total_elapsed = get_monotonic_time() - t_start
    avg_ocr_latency_ms = (
        sum(o.raw_ocr.processing_time_ms for o in all_raw_observations) / max(1, anpr.total_ocr_invocations)
    )

    # 4. Save results & debug crops
    cam_out_dir = Path(output_dir) / camera_id
    raw_dir = cam_out_dir / "raw_results"
    consensus_dir = cam_out_dir / "consensus"

    raw_dir.mkdir(parents=True, exist_ok=True)
    consensus_dir.mkdir(parents=True, exist_ok=True)

    raw_jsonl_path = str(raw_dir / "raw_observations.jsonl")
    consensus_json_path = str(consensus_dir / "anpr_consensus.json")

    eligible_count = sum(1 for el in all_eligibility_results if el.is_eligible)
    rejected_count = sum(1 for el in all_eligibility_results if not el.is_eligible)

    if save_results:
        with open(raw_jsonl_path, "w", encoding="utf-8") as f:
            for obs in all_raw_observations:
                f.write(json.dumps(obs.to_dict()) + "\n")

        summary_payload = {
            "camera_id": camera_id,
            "total_tracks_processed": len(candidates_by_track),
            "total_plate_candidates_loaded": total_candidates_read,
            "candidates_eligible_for_ocr": eligible_count,
            "candidates_rejected_before_ocr": rejected_count,
            "rejection_reasons": dict(rejection_counts),
            "total_ocr_invocations": anpr.total_ocr_invocations,
            "ocr_empty_count": anpr.total_ocr_empty,
            "ocr_low_confidence_count": anpr.total_ocr_low_conf,
            "ocr_valid_count": anpr.total_ocr_valid,
            "recognition_counts": counts,
            "confirmed_count": counts["CONFIRMED"],
            "probable_count": counts["PROBABLE"],
            "uncertain_count": counts["UNCERTAIN"],
            "unreadable_count": counts["UNREADABLE"],
            "avg_ocr_latency_ms": round(avg_ocr_latency_ms, 2),
            "tracks": [c.to_dict() for c in all_track_consensus],
        }

        with open(consensus_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    # Save debug crops if requested (Part 18)
    if save_debug_crops:
        debug_dir = cam_out_dir / "debug"
        for category in ["successful", "uncertain", "unreadable"]:
            (debug_dir / category).mkdir(parents=True, exist_ok=True)

        for c in all_track_consensus:
            cat = "successful" if c.recognition_status in ("CONFIRMED", "PROBABLE") else (
                "uncertain" if c.recognition_status == "UNCERTAIN" else "unreadable"
            )
            for ev in c.evidence_chain[:2]:
                src = ev.get("image_crop_path")
                if src and Path(src).exists():
                    dst = debug_dir / cat / Path(src).name
                    shutil.copy(src, dst)

    print("\n" + "=" * 65)
    print("  STEP 6 -- ANPR / OCR & CONSENSUS PERFORMANCE REPORT")
    print("=" * 65)
    print(f"  OCR Engine               : {anpr.ocr.engine_name} ({anpr.ocr.engine_version})")
    print(f"  Total candidates loaded  : {total_candidates_read}")
    print(f"  Eligible for OCR         : {eligible_count}")
    print(f"  Rejected before OCR      : {rejected_count}")
    for r_reason, r_count in sorted(rejection_counts.items()):
        print(f"    - {r_reason:<22}: {r_count}")
    print("-" * 65)
    print(f"  Total OCR invocations    : {anpr.total_ocr_invocations}")
    print(f"    - OCR empty            : {anpr.total_ocr_empty}")
    print(f"    - OCR low confidence   : {anpr.total_ocr_low_conf}")
    print(f"    - OCR valid text       : {anpr.total_ocr_valid}")
    print("-" * 65)
    print(f"  Tracks processed         : {len(candidates_by_track)}")
    print(f"  CONFIRMED plates         : {counts['CONFIRMED']}")
    print(f"  PROBABLE plates          : {counts['PROBABLE']}")
    print(f"  UNCERTAIN plates         : {counts['UNCERTAIN']}")
    print(f"  UNREADABLE plates        : {counts['UNREADABLE']}")
    print(f"  Avg OCR latency / crop   : {avg_ocr_latency_ms:.1f} ms")
    print(f"  Total pipeline time      : {t_total_elapsed:.2f} s")
    print(f"  Raw observations JSONL   : {raw_jsonl_path}")
    print(f"  Track consensus JSON     : {consensus_json_path}")
    print("=" * 65 + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Step 6 - Gujarat Police CCTV ANPR / OCR & Multi-Frame Consensus"
    )
    ap.add_argument("--camera-id", default=os.getenv("DEFAULT_CAMERA_ID", "cam01"),
                    help="Camera ID from normalized catalogue (default: cam01)")
    ap.add_argument("--input", default=None,
                    help="Path to Step 5 plate_candidates.jsonl (default: data/detections/<cam_id>/plate_candidates.jsonl)")
    ap.add_argument("--output", default="data/anpr",
                    help="Base output directory for ANPR (default: data/anpr)")
    ap.add_argument("--ocr-engine", default="easyocr", choices=["easyocr", "opencv"],
                    help="OCR engine (default: easyocr)")
    ap.add_argument("--min-ocr-confidence", type=float, default=0.02,
                    help="Minimum OCR engine confidence 0-1 (default: 0.02)")
    ap.add_argument("--min-consensus", type=float, default=0.05,
                    help="Minimum consensus score for CONFIRMED status (default: 0.05)")
    ap.add_argument("--min-probable", type=float, default=0.03,
                    help="Minimum consensus score for PROBABLE status (default: 0.03)")
    ap.add_argument("--min-confirmed-frames", type=int, default=1,
                    help="Minimum distinct frames with matching read for CONFIRMED status (default: 1)")
    ap.add_argument("--max-candidates-per-track", type=int, default=5,
                    help="Max plate candidate crops to process per track (default: 5)")
    ap.add_argument("--plate-format", default="auto", choices=["indian", "european", "auto"],
                    help="Plate format validator to use (default: auto). Use 'european' for synthetic/international footage.")
    ap.add_argument("--debug", action="store_true", help="Save representative debug crops")
    ap.add_argument("--no-display", action="store_true", help="Run in headless mode")
    ap.add_argument("--gpu", action="store_true", default=None, help="Force GPU mode for EasyOCR (default: auto-detect)")
    ap.add_argument("--cpu", action="store_true", default=False, help="Force CPU mode for EasyOCR")
    ap.add_argument("--save-results", action="store_true", default=True)

    args = ap.parse_args()

    input_file = args.input or f"data/detections/{args.camera_id}/plate_candidates.jsonl"

    use_gpu = False if args.cpu else (True if args.gpu else None)

    run_anpr_pipeline(
        camera_id=args.camera_id,
        input_jsonl_path=input_file,
        output_dir=args.output,
        min_ocr_confidence=args.min_ocr_confidence,
        min_consensus_score=args.min_consensus,
        min_probable_score=args.min_probable,
        min_confirmed_frames=args.min_confirmed_frames,
        max_candidates_per_track=args.max_candidates_per_track,
        save_results=args.save_results,
        save_debug_crops=args.debug,
        show_display=not args.no_display,
        engine_name=args.ocr_engine,
        use_gpu=use_gpu,
        plate_format=args.plate_format,
    )


if __name__ == "__main__":
    main()
