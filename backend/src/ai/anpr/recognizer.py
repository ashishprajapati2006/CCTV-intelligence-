"""ANPR Orchestrator for Step 6.

Orchestrates the entire Step 6 pipeline:
  Step 5 Plate Candidates -> Candidate Eligibility Filtering ->
  OCR Preprocessing Cascade -> Text Normalization & Audit ->
  Format Validation -> Evidence Scoring -> Multi-Frame Consensus ->
  Structured Output JSON/JSONL & Complete Evidence Chain.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2

from src.ai.anpr.schemas import ANPRObservation, TrackANPRConsensus, OCRResult
from src.ai.anpr.ocr import OCREngineFacade, OCRRecognizer
from src.ai.anpr.normalizer import normalize_with_audit
from src.ai.anpr.validator import validate_indian_plate
from src.ai.anpr.scoring import compute_observation_score
from src.ai.anpr.consensus import MultiFrameConsensusEngine
from src.ai.anpr.eligibility import CandidateEligibilityEvaluator, EligibilityResult
from src.common.logging import get_logger
from src.common.time import get_monotonic_time

logger = get_logger("anpr_recognizer")


class ANPRRecognizer:
    """High-level ANPR recognizer orchestrating OCR across variants and consensus across frames."""

    def __init__(
        self,
        ocr_engine: Optional[OCRRecognizer] = None,
        min_ocr_confidence: float = 0.20,
        consensus_engine: Optional[MultiFrameConsensusEngine] = None,
        eligibility_evaluator: Optional[CandidateEligibilityEvaluator] = None,
        enable_cascade: bool = True,
    ):
        self.ocr = ocr_engine or OCREngineFacade()
        self.min_ocr_confidence = min_ocr_confidence
        self.consensus_engine = consensus_engine or MultiFrameConsensusEngine()
        self.eligibility = eligibility_evaluator or CandidateEligibilityEvaluator()
        self.enable_cascade = enable_cascade

        # Diagnostic counters
        self.total_ocr_invocations = 0
        self.total_ocr_empty = 0
        self.total_ocr_low_conf = 0
        self.total_ocr_valid = 0

        logger.info(
            "ANPRRecognizer ready using OCR engine '%s' (%s)",
            self.ocr.engine_name, self.ocr.engine_version
        )

    def process_candidate_crop(
        self,
        candidate_dict: Dict[str, Any],
        image_path: str,
        variant_name: str,
    ) -> ANPRObservation:
        """Run OCR on a single crop variant image and return ANPRObservation."""
        resolved_path = Path(image_path)
        if not resolved_path.exists():
            resolved_path = Path.cwd() / image_path

        crop_img = cv2.imread(str(resolved_path)) if resolved_path.exists() else None

        # 1. OCR Inference (Always track invocation)
        self.total_ocr_invocations += 1

        if crop_img is None or crop_img.size == 0:
            logger.warning("[OCR ERROR] Unable to decode image at %s", image_path)
            raw_ocr = OCRResult(
                raw_text="", confidence=0.0, character_confidences=None,
                processing_time_ms=0.0, engine_name=self.ocr.engine_name,
                engine_version=self.ocr.engine_version, variant_name=variant_name
            )
        else:
            raw_ocr = self.ocr.recognize(crop_img, variant_name=variant_name)

        if not raw_ocr.raw_text:
            self.total_ocr_empty += 1
        elif raw_ocr.confidence < self.min_ocr_confidence:
            self.total_ocr_low_conf += 1
        else:
            self.total_ocr_valid += 1

        # 2. Text Normalization
        norm_ocr = normalize_with_audit(raw_ocr.raw_text, apply_contextual_corrections=True)

        # 3. Indian Format Validation
        is_valid_fmt, fmt_name, _ = validate_indian_plate(norm_ocr.normalized_text)
        norm_ocr.is_valid_format = is_valid_fmt
        norm_ocr.format_name = fmt_name

        # 4. Evidence Scoring
        det_conf = float(candidate_dict.get("plate_confidence", 0.80))
        qual_score = float(candidate_dict.get("plate_quality_score", 0.70))
        obs_score = compute_observation_score(raw_ocr, norm_ocr, det_conf, qual_score)

        return ANPRObservation(
            camera_id=candidate_dict["camera_id"],
            track_id=candidate_dict["track_id"],
            frame_id=int(candidate_dict["frame_id"]),
            pts_ms=candidate_dict.get("pts_ms"),
            has_valid_pts=bool(candidate_dict.get("has_valid_pts", True)),
            local_receive_monotonic=float(candidate_dict.get("local_receive_monotonic", get_monotonic_time())),
            plate_bbox=list(candidate_dict.get("plate_bbox", [0, 0, 0, 0])),
            plate_detection_confidence=det_conf,
            plate_quality_score=qual_score,
            variant_name=variant_name,
            raw_ocr=raw_ocr,
            normalized_ocr=norm_ocr,
            observation_score=obs_score,
            image_crop_path=str(resolved_path),
        )

    def process_plate_candidate(
        self,
        candidate_dict: Dict[str, Any]
    ) -> Tuple[List[ANPRObservation], EligibilityResult]:
        """Process one Step 5 PlateCandidate record using the Preprocessing Cascade."""
        # 1. Eligibility evaluation
        elig = self.eligibility.evaluate_candidate(candidate_dict)
        if not elig.is_eligible:
            return [], elig

        observations = []
        orig_path = candidate_dict.get("original_crop_path")

        # Stage 1: Try Original
        if orig_path:
            obs = self.process_candidate_crop(candidate_dict, orig_path, "original")
            observations.append(obs)

            # If original succeeded with valid format & high confidence, cascade is complete
            if obs.normalized_ocr.is_valid_format and obs.raw_ocr.confidence >= 0.75:
                return observations, elig

        # Preprocessing Cascade Stages (if enabled and original was empty or weak)
        if self.enable_cascade:
            proc_paths = candidate_dict.get("processed_crop_paths", {})
            cascade_stages = ["grayscale", "upscaled", "contrast", "sharpened"]

            for stage in cascade_stages:
                stage_path = proc_paths.get(stage)
                if stage_path:
                    stage_obs = self.process_candidate_crop(candidate_dict, stage_path, stage)
                    observations.append(stage_obs)

                    # Stop cascade early if high-confidence valid reading reached
                    if stage_obs.normalized_ocr.is_valid_format and stage_obs.raw_ocr.confidence >= 0.75:
                        break

        return observations, elig

    def process_track_candidates(
        self,
        camera_id: str,
        track_id: str,
        track_candidates: List[Dict[str, Any]],
    ) -> Tuple[TrackANPRConsensus, List[ANPRObservation], List[EligibilityResult]]:
        """Process all Step 5 plate candidates for one vehicle track and return consensus."""
        all_obs: List[ANPRObservation] = []
        all_eligs: List[EligibilityResult] = []

        for cand_dict in track_candidates:
            c_obs, elig = self.process_plate_candidate(cand_dict)
            all_eligs.append(elig)
            all_obs.extend(c_obs)

        consensus = self.consensus_engine.resolve_track_consensus(
            camera_id=camera_id,
            track_id=track_id,
            observations=all_obs,
        )

        return consensus, all_obs, all_eligs
