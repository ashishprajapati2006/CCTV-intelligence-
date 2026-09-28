#!/usr/bin/env python3
"""Step 6 - Comprehensive ANPR Evaluation & Diagnostic Tool.

Evaluates an ANPR recognition model / pipeline against labeled development
or held-out datasets with complete condition, source, track, and failure analysis.

Usage:
  # Evaluate development set:
  python scripts/evaluate_anpr.py --dataset data/evaluation/anpr/development

  # Evaluate held-out set:
  python scripts/evaluate_anpr.py --dataset data/evaluation/anpr/heldout

  # Run preprocessing ablation:
  python scripts/evaluate_anpr.py --dataset data/evaluation/anpr/development --ablation

  # Run confidence threshold analysis:
  python scripts/evaluate_anpr.py --dataset data/evaluation/anpr/development --threshold-analysis

Key Capabilities:
  1. Dataset Sample Validation (excludes missing/corrupt samples from metrics).
  2. Image-Level Metrics (exact match, normalized match, character accuracy).
  3. Track-Level Multi-Frame Consensus Metrics (distinguishes image vs track performance).
  4. Condition-Wise Reporting with explicit sample counts (no bare 100% without sample count).
  5. Source Breakdown (sentinel vs controlled).
  6. Failure Classification (14 explicit categories) and False Recognition export.
  7. Preprocessing Ablation Comparison (original, grayscale, upscaled, contrast, sharpened).
  8. Confidence Threshold Tradeoff Analysis (0.20 to 0.70).
  9. Human-readable (report.txt) and JSON (report.json) output generation.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai.anpr.consensus import MultiFrameConsensusEngine
from src.ai.anpr.evaluation import (
    ALLOWED_CONDITIONS,
    FAILURE_CATEGORIES,
    DatasetSampleValidation,
    SampleDiagnostic,
    classify_failure,
    compute_character_accuracy,
    validate_dataset_sample,
)
from src.ai.anpr.normalizer import normalize_raw_text, normalize_with_audit
from src.ai.anpr.ocr import OCREngineFacade
from src.ai.plate_preprocessor import generate_preprocessing_variants
from src.ai.anpr.schemas import ANPRObservation, OCRResult
from src.ai.anpr.scoring import compute_observation_score
from src.ai.anpr.validator import validate_indian_plate
from src.common.logging import get_logger

logger = get_logger("evaluate_anpr")


def resolve_dataset_paths(dataset_arg: str) -> Tuple[Path, Path, str]:
    """Resolve dataset directory, labels file, and split name."""
    p = Path(dataset_arg)
    if p.is_dir():
        split_name = p.name.upper()
        labels_file = p / "labels.json"
        dataset_root = p
    elif p.is_file():
        split_name = p.parent.name.upper()
        labels_file = p
        dataset_root = p.parent
    else:
        split_name = "CUSTOM"
        labels_file = p
        dataset_root = p.parent

    return dataset_root, labels_file, split_name


def evaluate_anpr_dataset(
    dataset_path: str,
    output_dir: Optional[str] = None,
    engine_name: str = "easyocr",
    min_confidence: float = 0.20,
    run_ablation: bool = False,
    run_thresholds: bool = False,
) -> Dict[str, Any]:
    dataset_root, labels_file, split_name = resolve_dataset_paths(dataset_path)

    if not labels_file.exists():
        logger.error("Evaluation dataset labels file not found: %s", str(labels_file))
        sys.exit(1)

    with open(labels_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    results_dir = Path(output_dir or f"data/evaluation/anpr/results/{split_name.lower()}")
    false_rec_dir = results_dir / "false_recognitions"
    results_dir.mkdir(parents=True, exist_ok=True)
    false_rec_dir.mkdir(parents=True, exist_ok=True)

    header_title = f"GUJARAT POLICE CCTV - ANPR EVALUATION ({split_name} SET)"
    print("=" * 75)
    print(f"  {header_title}")
    print("=" * 75)
    print(f"  Dataset Path     : {labels_file}")
    print(f"  Split Mode       : DATASET: {split_name}")
    print(f"  Total Records    : {len(records)}")
    print(f"  OCR Engine       : {engine_name}")
    print(f"  Confidence Cutoff: >= {min_confidence}")
    print("=" * 75 + "\n")

    # 1. Sample Validation Step
    validated_samples: List[Tuple[DatasetSampleValidation, Optional[np.ndarray], Dict[str, Any]]] = []
    validation_counts = defaultdict(int)

    for rec in records:
        val, img = validate_dataset_sample(rec, dataset_root=dataset_root)
        validated_samples.append((val, img, rec))
        validation_counts[val.status] += 1

    valid_samples = [(v, img, r) for v, img, r in validated_samples if v.is_valid]
    invalid_samples = [(v, img, r) for v, img, r in validated_samples if not v.is_valid]

    print("[*] Dataset Validation Summary:")
    print(f"    - VALID samples        : {len(valid_samples)}")
    print(f"    - MISSING / INVALID    : {len(invalid_samples)}")
    for status_name, count in validation_counts.items():
        if status_name != "VALID":
            print(f"      * {status_name:<20}: {count}")
    print("-" * 75 + "\n")

    # 2. OCR Inference on Valid Samples
    ocr = OCREngineFacade(preferred_engine=engine_name)
    diagnostics: List[SampleDiagnostic] = []
    condition_stats = defaultdict(lambda: {"total": 0, "exact": 0, "norm": 0, "char_acc": 0.0})
    source_stats = defaultdict(lambda: {"total": 0, "exact": 0, "norm": 0, "char_acc": 0.0})
    failure_counts = defaultdict(int)

    # Track-level observations grouping: (camera_id, track_id) -> list of observations
    track_observations = defaultdict(list)
    track_ground_truths = {}

    total_exact_matches = 0
    total_norm_matches = 0
    total_char_accuracy = 0.0
    total_unreadable = 0
    total_false_recognitions = 0
    total_ocr_latency_ms = 0.0

    print(f"{'IMAGE':<32} {'DIM':<9} {'COND':<12} {'SRC':<10} {'GT':<12} {'OCR':<12} {'STATUS'}")
    print("-" * 105)

    for val, img, rec in valid_samples:
        t0 = time.monotonic()
        raw_res = ocr.recognize(img, variant_name="original")
        lat_ms = (time.monotonic() - t0) * 1000.0
        total_ocr_latency_ms += lat_ms

        raw_text = raw_res.raw_text
        ocr_conf = raw_res.confidence

        # Audited normalization
        norm_res = normalize_with_audit(raw_text, apply_contextual_corrections=True)
        norm_text = norm_res.normalized_text
        gt_norm = val.normalized_ground_truth

        # Accuracy checks
        exact = (raw_text == val.ground_truth)
        norm_match = (norm_text == gt_norm and bool(norm_text))
        char_acc = compute_character_accuracy(gt_norm, norm_text)

        total_char_accuracy += char_acc
        if exact:
            total_exact_matches += 1
        if norm_match:
            total_norm_matches += 1
        elif not norm_text:
            total_unreadable += 1
        else:
            total_false_recognitions += 1

        # Classify Failure
        failure_reason = classify_failure(
            ground_truth=val.ground_truth,
            raw_ocr=raw_text,
            normalized_ocr=norm_text,
            ocr_conf=ocr_conf,
            val=val,
            min_confidence=min_confidence,
        )
        failure_counts[failure_reason] += 1

        # Diagnostic record
        diag = SampleDiagnostic(
            image_path=val.image_path,
            camera_id=str(rec.get("camera_id", "cam01")),
            track_id=str(rec.get("track_id", "TRK-0001")),
            condition=val.condition,
            source=val.source,
            ground_truth=val.ground_truth,
            normalized_ground_truth=gt_norm,
            raw_ocr=raw_text,
            normalized_ocr=norm_text,
            ocr_confidence=ocr_conf,
            plate_detection_confidence=float(rec.get("plate_confidence", 0.85) or 0.85),
            plate_quality_score=float(rec.get("plate_quality_score", 0.70) or 0.70),
            image_width=val.width,
            image_height=val.height,
            exact_match=exact,
            normalized_match=norm_match,
            character_accuracy=round(char_acc, 4),
            failure_reason=failure_reason,
            processing_time_ms=round(lat_ms, 2),
        )
        diagnostics.append(diag)

        # Condition & Source stats
        for stat_dict, key in [(condition_stats, val.condition), (source_stats, val.source)]:
            stat_dict[key]["total"] += 1
            if exact:
                stat_dict[key]["exact"] += 1
            if norm_match:
                stat_dict[key]["norm"] += 1
            stat_dict[key]["char_acc"] += char_acc

        # Group for Track-Level Multi-Frame Evaluation
        trk_key = (diag.camera_id, diag.track_id)
        track_ground_truths[trk_key] = gt_norm

        # Build ANPRObservation for consensus engine
        obs_score = compute_observation_score(
            raw_res, norm_res, diag.plate_detection_confidence, diag.plate_quality_score
        )
        anpr_obs = ANPRObservation(
            camera_id=diag.camera_id,
            track_id=diag.track_id,
            frame_id=int(rec.get("frame_id", 1)),
            pts_ms=float(rec.get("pts_ms", 1000.0) or 1000.0),
            has_valid_pts=True,
            local_receive_monotonic=time.monotonic(),
            plate_bbox=[0, 0, val.width, val.height],
            plate_detection_confidence=diag.plate_detection_confidence,
            plate_quality_score=diag.plate_quality_score,
            variant_name="original",
            raw_ocr=raw_res,
            normalized_ocr=norm_res,
            observation_score=obs_score,
            image_crop_path=val.image_path,
        )
        track_observations[trk_key].append(anpr_obs)

        # Export false recognition sample if failure is false recognition or confusion
        if failure_reason in ("FALSE_RECOGNITION", "CHARACTER_CONFUSION", "GLARE", "BLUR"):
            sample_dest = false_rec_dir / Path(val.image_path).name
            if not sample_dest.exists():
                shutil.copy(val.image_path, sample_dest)

        status_tag = "CORRECT" if norm_match else failure_reason
        short_img = Path(val.image_path).name[:30]
        dim_str = f"{val.width}x{val.height}"
        print(f"{short_img:<32} {dim_str:<9} {val.condition:<12} {val.source:<10} {val.ground_truth:<12} {norm_text:<12} {status_tag}")

    print("-" * 105 + "\n")

    # 3. Calculate Image-Level Metrics
    N = len(valid_samples)
    img_exact_acc = (total_exact_matches / float(N) * 100.0) if N else 0.0
    img_norm_acc = (total_norm_matches / float(N) * 100.0) if N else 0.0
    img_avg_char_acc = (total_char_accuracy / float(N) * 100.0) if N else 0.0
    img_unreadable_rate = (total_unreadable / float(N) * 100.0) if N else 0.0
    img_false_rec_rate = (total_false_recognitions / float(N) * 100.0) if N else 0.0
    avg_latency = (total_ocr_latency_ms / float(N)) if N else 0.0

    # 4. Calculate Track-Level Metrics
    consensus_engine = MultiFrameConsensusEngine(min_confirmed_score=0.65, min_confirmed_frames=1)
    track_exact_matches = 0
    track_norm_matches = 0
    track_uncertain_count = 0
    track_total_consensus_score = 0.0
    track_total_supporting_frames = 0
    track_diagnostics = []

    for trk_key, obs_list in track_observations.items():
        cam_id, trk_id = trk_key
        gt_trk = track_ground_truths[trk_key]

        c_res = consensus_engine.resolve_track_consensus(
            camera_id=cam_id,
            track_id=trk_id,
            observations=obs_list,
        )

        pred_trk = c_res.final_registration_number or ""
        trk_norm_match = (normalize_raw_text(pred_trk) == gt_trk and bool(pred_trk))
        if pred_trk == gt_trk and bool(pred_trk):
            track_exact_matches += 1
        if trk_norm_match:
            track_norm_matches += 1

        if c_res.recognition_status in ("UNCERTAIN", "UNREADABLE"):
            track_uncertain_count += 1

        track_total_consensus_score += c_res.consensus_score
        track_total_supporting_frames += c_res.supporting_frame_count

        track_diagnostics.append({
            "camera_id": cam_id,
            "track_id": trk_id,
            "ground_truth": gt_trk,
            "recognized_number": pred_trk,
            "status": c_res.recognition_status,
            "consensus_score": round(c_res.consensus_score, 4),
            "supporting_frames": c_res.supporting_frame_count,
            "observations_count": len(obs_list),
            "is_correct": trk_norm_match,
        })

    num_tracks = len(track_observations)
    trk_exact_acc = (track_exact_matches / float(num_tracks) * 100.0) if num_tracks else 0.0
    trk_norm_acc = (track_norm_matches / float(num_tracks) * 100.0) if num_tracks else 0.0
    trk_uncertain_rate = (track_uncertain_count / float(num_tracks) * 100.0) if num_tracks else 0.0
    trk_avg_consensus = (track_total_consensus_score / float(num_tracks)) if num_tracks else 0.0
    trk_avg_supporting = (track_total_supporting_frames / float(num_tracks)) if num_tracks else 0.0

    # 5. Preprocessing Ablation (Optional)
    ablation_results = {}
    if run_ablation and valid_samples:
        print("[*] Running Preprocessing Ablation Analysis across variants...")
        variants_list = ["original", "grayscale", "upscaled", "contrast", "sharpened"]

        for var_name in variants_list:
            var_norm_hits = 0
            var_char_acc = 0.0
            for val, img, rec in valid_samples:
                variant_img = img
                if var_name != "original":
                    p_variants = generate_preprocessing_variants(img)
                    variant_img = p_variants.get(var_name, img)

                res = ocr.recognize(variant_img, variant_name=var_name)
                norm_v = normalize_with_audit(res.raw_text).normalized_text
                if norm_v == val.normalized_ground_truth and norm_v:
                    var_norm_hits += 1
                var_char_acc += compute_character_accuracy(val.normalized_ground_truth, norm_v)

            ablation_results[var_name] = {
                "normalized_exact_pct": round((var_norm_hits / float(N)) * 100.0, 2),
                "avg_char_acc_pct": round((var_char_acc / float(N)) * 100.0, 2),
            }

    # 6. Confidence Threshold Tradeoff Analysis (Optional)
    threshold_tradeoffs = {}
    if run_thresholds and diagnostics:
        print("[*] Running Confidence Threshold Tradeoff Analysis...")
        cutoffs = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
        for cut in cutoffs:
            t_confirmed = 0
            t_probable = 0
            t_uncertain = 0
            t_unreadable = 0
            t_false_rec = 0

            for d in diagnostics:
                if not d.normalized_ocr:
                    t_unreadable += 1
                elif d.ocr_confidence < cut:
                    t_uncertain += 1
                elif d.normalized_match:
                    if d.ocr_confidence >= 0.60:
                        t_confirmed += 1
                    else:
                        t_probable += 1
                else:
                    t_false_rec += 1

            threshold_tradeoffs[f"conf_cutoff_{cut:.2f}"] = {
                "confirmed": t_confirmed,
                "probable": t_probable,
                "uncertain": t_uncertain,
                "unreadable": t_unreadable,
                "false_recognition": t_false_rec,
            }

    # 7. Print Performance Reports
    report_lines = []
    def p(line: str = ""):
        print(line)
        report_lines.append(line)

    p("=" * 75)
    p(f"  ANPR EVALUATION PERFORMANCE REPORT: {split_name} SET")
    p("=" * 75)
    p(f"  Dataset split                  : {split_name}")
    p(f"  Total records labeled          : {len(records)}")
    p(f"  Valid image samples            : {N}")
    p(f"  Invalid / missing samples      : {len(invalid_samples)}")
    p(f"  OCR Engine                     : {ocr.engine_name} ({ocr.engine_version})")
    p(f"  Average OCR latency            : {avg_latency:.1f} ms / crop")
    p("-" * 75)
    p("  IMAGE-LEVEL PERFORMANCE:")
    p(f"    - Exact-Match Accuracy       : {img_exact_acc:5.1f}% ({total_exact_matches}/{N})")
    p(f"    - Normalized Exact-Match     : {img_norm_acc:5.1f}% ({total_norm_matches}/{N})")
    p(f"    - Average Character Accuracy : {img_avg_char_acc:5.1f}%")
    p(f"    - Unreadable Rate            : {img_unreadable_rate:5.1f}% ({total_unreadable}/{N})")
    p(f"    - False Recognition Rate     : {img_false_rec_rate:5.1f}% ({total_false_recognitions}/{N})")
    p("-" * 75)
    p("  TRACK-LEVEL MULTI-FRAME PERFORMANCE:")
    p(f"    - Total vehicle tracks       : {num_tracks}")
    p(f"    - Track Normalized Accuracy  : {trk_norm_acc:5.1f}% ({track_norm_matches}/{num_tracks})")
    p(f"    - Track Exact Accuracy       : {trk_exact_acc:5.1f}% ({track_exact_matches}/{num_tracks})")
    p(f"    - Track Uncertainty Rate     : {trk_uncertain_rate:5.1f}% ({track_uncertain_count}/{num_tracks})")
    p(f"    - Average Supporting Frames  : {trk_avg_supporting:4.2f}")
    p(f"    - Average Consensus Score    : {trk_avg_consensus:4.2f}")
    p("-" * 75)
    p("  ACCURACY BY SCENE CONDITION (with explicit sample counts):")
    for cond_name, cdata in sorted(condition_stats.items()):
        c_tot = cdata["total"]
        c_norm = cdata["norm"]
        c_pct = (c_norm / float(c_tot) * 100.0) if c_tot else 0.0
        insufficient_tag = " [insufficient sample count for statistical significance]" if c_tot <= 2 else ""
        p(f"    - {cond_name:<14}: {c_pct:5.1f}% normalized match ({c_norm}/{c_tot}){insufficient_tag}")
    p("-" * 75)
    p("  ACCURACY BY DATA SOURCE PROVENANCE:")
    for src_name, sdata in sorted(source_stats.items()):
        s_tot = sdata["total"]
        s_norm = sdata["norm"]
        s_pct = (s_norm / float(s_tot) * 100.0) if s_tot else 0.0
        p(f"    - {src_name:<14}: {s_pct:5.1f}% normalized match ({s_norm}/{s_tot})")
    p("-" * 75)
    p("  FAILURE REASON BREAKDOWN:")
    for r_name in FAILURE_CATEGORIES:
        count = failure_counts.get(r_name, 0)
        if count > 0 or r_name == "CORRECT":
            p(f"    - {r_name:<22}: {count}")

    if ablation_results:
        p("-" * 75)
        p("  PREPROCESSING ABLATION ANALYSIS:")
        for v_name, v_res in ablation_results.items():
            p(f"    - {v_name:<14}: {v_res['normalized_exact_pct']:5.1f}% norm match | {v_res['avg_char_acc_pct']:5.1f}% char acc")

    if threshold_tradeoffs:
        p("-" * 75)
        p("  CONFIDENCE THRESHOLD TRADEOFF TABLE:")
        p(f"    {'Cutoff':<12} {'Confirmed':<11} {'Probable':<10} {'Uncertain':<11} {'Unreadable':<12} {'False Rec':<10}")
        for t_name, t_row in threshold_tradeoffs.items():
            c_val = t_name.replace("conf_cutoff_", "")
            p(f"    {c_val:<12} {t_row['confirmed']:<11} {t_row['probable']:<10} {t_row['uncertain']:<11} {t_row['unreadable']:<12} {t_row['false_recognition']:<10}")

    p("=" * 75 + "\n")

    # 8. Save structured JSON and text report
    metrics_payload = {
        "dataset_split": split_name,
        "dataset_path": str(labels_file),
        "total_labeled_records": len(records),
        "valid_samples_count": N,
        "invalid_samples_count": len(invalid_samples),
        "image_level_metrics": {
            "exact_match_accuracy_pct": round(img_exact_acc, 2),
            "normalized_exact_match_pct": round(img_norm_acc, 2),
            "avg_character_accuracy_pct": round(img_avg_char_acc, 2),
            "unreadable_rate_pct": round(img_unreadable_rate, 2),
            "false_recognition_rate_pct": round(img_false_rec_rate, 2),
            "average_ocr_latency_ms": round(avg_latency, 2),
        },
        "track_level_metrics": {
            "total_tracks": num_tracks,
            "track_normalized_accuracy_pct": round(trk_norm_acc, 2),
            "track_exact_accuracy_pct": round(trk_exact_acc, 2),
            "track_uncertainty_rate_pct": round(trk_uncertain_rate, 2),
            "avg_supporting_frames": round(trk_avg_supporting, 2),
            "avg_consensus_score": round(trk_avg_consensus, 4),
            "tracks": track_diagnostics,
        },
        "condition_breakdown": {
            k: {
                "total": v["total"],
                "normalized_correct": v["norm"],
                "normalized_accuracy_pct": round((v["norm"] / float(v["total"]) * 100.0) if v["total"] else 0.0, 2),
                "is_statistically_sufficient": v["total"] > 2,
            }
            for k, v in condition_stats.items()
        },
        "source_breakdown": {
            k: {
                "total": v["total"],
                "normalized_correct": v["norm"],
                "normalized_accuracy_pct": round((v["norm"] / float(v["total"]) * 100.0) if v["total"] else 0.0, 2),
            }
            for k, v in source_stats.items()
        },
        "failure_breakdown": dict(failure_counts),
        "ablation_results": ablation_results,
        "threshold_tradeoffs": threshold_tradeoffs,
        "per_sample_diagnostics": [d.to_dict() for d in diagnostics],
    }

    report_json_path = results_dir / "report.json"
    report_txt_path = results_dir / "report.txt"

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info("Saved evaluation report to: %s and %s", report_json_path, report_txt_path)
    return metrics_payload


def main():
    ap = argparse.ArgumentParser(description="Step 6 - Comprehensive ANPR Evaluation Tool")
    ap.add_argument("--dataset", default="data/evaluation/anpr/development",
                    help="Path to dataset dir or labels.json (default: data/evaluation/anpr/development)")
    ap.add_argument("--output", default=None, help="Custom output directory for reports")
    ap.add_argument("--ocr-engine", default="easyocr", choices=["easyocr", "opencv"])
    ap.add_argument("--min-confidence", type=float, default=0.20)
    ap.add_argument("--ablation", action="store_true", help="Run preprocessing ablation analysis")
    ap.add_argument("--threshold-analysis", action="store_true", help="Run confidence threshold tradeoff analysis")
    args = ap.parse_args()

    evaluate_anpr_dataset(
        dataset_path=args.dataset,
        output_dir=args.output,
        engine_name=args.ocr_engine,
        min_confidence=args.min_confidence,
        run_ablation=args.ablation,
        run_thresholds=args.threshold_analysis,
    )


if __name__ == "__main__":
    main()
