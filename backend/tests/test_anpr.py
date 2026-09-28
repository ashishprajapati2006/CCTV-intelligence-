"""Unit tests for Step 6 — ANPR / OCR & Multi-Frame Consensus.

All tests use synthetic OCR readings and mock images.
No live stream or external GPU/network required.
"""
from __future__ import annotations

import time
import pytest
from pathlib import Path

from src.ai.anpr.schemas import (
    OCRResult,
    NormalizedOCRResult,
    ANPRObservation,
    TrackANPRConsensus,
)
from src.ai.anpr.normalizer import normalize_raw_text, normalize_with_audit
from src.ai.anpr.validator import validate_indian_plate
from src.ai.anpr.scoring import compute_observation_score
from src.ai.anpr.consensus import MultiFrameConsensusEngine, _levenshtein_distance
from src.ai.anpr.eligibility import CandidateEligibilityEvaluator


# ════════════════════════════════════════════════════════════════════════════
#  1. Text Normalization & Audit Logs
# ════════════════════════════════════════════════════════════════════════════

class TestTextNormalization:
    def test_strip_spaces_hyphens_lowercase(self):
        text = " gj - 01  ab- 1234 "
        norm = normalize_raw_text(text)
        assert norm == "GJ01AB1234"

    def test_empty_string_normalization(self):
        assert normalize_raw_text("") == ""
        assert normalize_raw_text(None) == ""

    def test_contextual_correction_audit_log(self):
        # 'O' in numeric position (2, 3) -> should audit O -> 0
        raw = "GJO1AB1234"  # 'O' at index 2
        norm = normalize_with_audit(raw, apply_contextual_corrections=True)
        assert norm.normalized_text == "GJ01AB1234"
        assert norm.has_corrections is True
        assert len(norm.corrections) == 1
        assert norm.corrections[0].original_char == "O"
        assert norm.corrections[0].corrected_char == "0"

    def test_ind_country_badge_strip(self):
        raw = "INDGJ01AB1234"
        norm = normalize_with_audit(raw, apply_contextual_corrections=True)
        assert norm.normalized_text == "GJ01AB1234"
        assert any(c.reason == "strip_ind_country_badge_prefix" for c in norm.corrections)


# ════════════════════════════════════════════════════════════════════════════
#  2. Indian Registration Plate Validation
# ════════════════════════════════════════════════════════════════════════════

class TestIndianPlateValidation:
    def test_standard_10_char_format(self):
        valid, fmt, boost = validate_indian_plate("GJ01AB1234")
        assert valid is True
        assert fmt == "standard_indian_10"
        assert boost == pytest.approx(1.0)

    def test_short_legacy_format(self):
        valid, fmt, boost = validate_indian_plate("GJ05A1234")
        assert valid is True
        assert fmt == "short_indian_legacy"

    def test_bharat_series_bh(self):
        valid, fmt, boost = validate_indian_plate("22BH1234AA")
        assert valid is True
        assert fmt == "bharat_series_bh"

    def test_incomplete_text_no_hallucination(self):
        """CRITICAL: Short/incomplete string must return invalid, NOT hallucinated text."""
        valid, fmt, boost = validate_indian_plate("GJ01AB12")
        assert valid is False
        assert fmt == "unrecognized_format"

    def test_invalid_random_string(self):
        valid, fmt, boost = validate_indian_plate("HELLO123WORLD")
        assert valid is False


# ════════════════════════════════════════════════════════════════════════════
#  3. Candidate Eligibility & Rejection Diagnostics
# ════════════════════════════════════════════════════════════════════════════

class TestCandidateEligibility:
    def setup_method(self):
        self.evaluator = CandidateEligibilityEvaluator(
            min_plate_width=20,
            min_plate_height=8,
            min_plate_area=160,
            min_detection_confidence=0.30,
            min_quality_score=0.20,
        )

    def test_eligible_candidate(self, tmp_path):
        dummy_img = tmp_path / "crop.jpg"
        dummy_img.write_text("fake")
        cand = {
            "candidate_id": "1",
            "track_id": "TRK-001",
            "frame_id": 10,
            "original_crop_path": str(dummy_img),
            "plate_bbox": [10, 10, 60, 30],  # 50x20 = 1000 area
            "plate_confidence": 0.85,
            "plate_quality_score": 0.75,
        }
        res = self.evaluator.evaluate_candidate(cand)
        assert res.is_eligible is True
        assert res.rejection_reason == "NONE"

    def test_too_small_crop_rejected(self, tmp_path):
        dummy_img = tmp_path / "crop.jpg"
        dummy_img.write_text("fake")
        cand = {
            "candidate_id": "1",
            "track_id": "TRK-001",
            "frame_id": 10,
            "original_crop_path": str(dummy_img),
            "plate_bbox": [10, 10, 15, 15],  # 5x5 = 25 area (< 160)
            "plate_confidence": 0.85,
            "plate_quality_score": 0.75,
        }
        res = self.evaluator.evaluate_candidate(cand)
        assert res.is_eligible is False
        assert res.rejection_reason == "TOO_SMALL"

    def test_missing_crop_rejected(self):
        cand = {
            "candidate_id": "1",
            "track_id": "TRK-001",
            "frame_id": 10,
            "original_crop_path": "non_existent_file_12345.jpg",
            "plate_bbox": [10, 10, 60, 30],
            "plate_confidence": 0.85,
            "plate_quality_score": 0.75,
        }
        res = self.evaluator.evaluate_candidate(cand)
        assert res.is_eligible is False
        assert res.rejection_reason == "MISSING_CROP"

    def test_low_detection_confidence_rejected(self, tmp_path):
        dummy_img = tmp_path / "crop.jpg"
        dummy_img.write_text("fake")
        cand = {
            "candidate_id": "1",
            "track_id": "TRK-001",
            "frame_id": 10,
            "original_crop_path": str(dummy_img),
            "plate_bbox": [10, 10, 60, 30],
            "plate_confidence": 0.15,  # < 0.30
            "plate_quality_score": 0.75,
        }
        res = self.evaluator.evaluate_candidate(cand)
        assert res.is_eligible is False
        assert res.rejection_reason == "LOW_DETECTION_CONFIDENCE"

    def test_invalid_bbox_rejected(self, tmp_path):
        dummy_img = tmp_path / "crop.jpg"
        dummy_img.write_text("fake")
        cand = {
            "candidate_id": "1",
            "track_id": "TRK-001",
            "frame_id": 10,
            "original_crop_path": str(dummy_img),
            "plate_bbox": [60, 30, 10, 10],  # inverted coordinates
            "plate_confidence": 0.85,
            "plate_quality_score": 0.75,
        }
        res = self.evaluator.evaluate_candidate(cand)
        assert res.is_eligible is False
        assert res.rejection_reason == "INVALID_BBOX"


# ════════════════════════════════════════════════════════════════════════════
#  4. Evidence Scoring & Independent Sub-Metrics
# ════════════════════════════════════════════════════════════════════════════

class TestEvidenceScoring:
    def test_compute_observation_score(self):
        raw_ocr = OCRResult(
            raw_text="GJ01AB1234", confidence=0.90, character_confidences=[0.9]*10,
            processing_time_ms=15.0, engine_name="test", engine_version="1.0", variant_name="original"
        )
        norm_ocr = NormalizedOCRResult("GJ01AB1234", "GJ01AB1234", is_valid_format=True, format_name="standard_indian_10")
        score = compute_observation_score(raw_ocr, norm_ocr, plate_detection_confidence=0.95, plate_quality_score=0.85)

        assert 0.0 <= score <= 1.0
        assert score > 0.80

    def test_independent_metrics_preserved_in_schema(self):
        obs = ANPRObservation(
            camera_id="cam01", track_id="TRK-001", frame_id=1, pts_ms=1000.0, has_valid_pts=True,
            local_receive_monotonic=100.0, plate_bbox=[10, 10, 100, 40], plate_detection_confidence=0.92,
            plate_quality_score=0.88, variant_name="original",
            raw_ocr=OCRResult("GJ01AB1234", 0.90, None, 10.0, "test", "1.0", "original"),
            normalized_ocr=NormalizedOCRResult("GJ01AB1234", "GJ01AB1234", is_valid_format=True),
            observation_score=0.90,
        )
        odict = obs.to_dict()
        assert odict["plate_detection_confidence"] == pytest.approx(0.92)
        assert odict["plate_quality_score"] == pytest.approx(0.88)
        assert odict["raw_ocr"]["confidence"] == pytest.approx(0.90)


# ════════════════════════════════════════════════════════════════════════════
#  5. Levenshtein Distance & Fuzzy Match
# ════════════════════════════════════════════════════════════════════════════

class TestFuzzyMatch:
    def test_levenshtein_distance(self):
        assert _levenshtein_distance("GJ01AB1234", "GJ01AB1234") == 0
        assert _levenshtein_distance("GJ01AB1234", "GJ01A81234") == 1 # B vs 8
        assert _levenshtein_distance("GJ01AB1234", "GJ02CD5678") > 3


# ════════════════════════════════════════════════════════════════════════════
#  6. Multi-Frame Consensus Engine & Recognition States
# ════════════════════════════════════════════════════════════════════════════

class TestConsensusEngine:
    def _make_obs(
        self,
        frame_id: int,
        pts_ms: float,
        text: str,
        score: float = 0.85,
        variant: str = "original",
        is_valid_fmt: bool = True,
    ) -> ANPRObservation:
        return ANPRObservation(
            camera_id="cam01", track_id="TRK-001", frame_id=frame_id, pts_ms=pts_ms, has_valid_pts=True,
            local_receive_monotonic=time.monotonic(), plate_bbox=[10, 10, 100, 40],
            plate_detection_confidence=0.90, plate_quality_score=0.85, variant_name=variant,
            raw_ocr=OCRResult(text, score, None, 10.0, "test", "1.0", variant),
            normalized_ocr=NormalizedOCRResult(text, text, is_valid_format=is_valid_fmt, format_name="standard_indian_10" if is_valid_fmt else None),
            observation_score=score,
        )

    def test_confirmed_status_multi_frame_agreement(self):
        engine = MultiFrameConsensusEngine(min_confirmed_score=0.70, min_confirmed_frames=2)
        obs = [
            self._make_obs(frame_id=1, pts_ms=1000.0, text="GJ01AB1234"),
            self._make_obs(frame_id=2, pts_ms=1080.0, text="GJ01AB1234"),
            self._make_obs(frame_id=3, pts_ms=1160.0, text="GJ01AB1234"),
        ]
        res = engine.resolve_track_consensus("cam01", "TRK-001", obs)
        assert res.recognition_status == "CONFIRMED"
        assert res.final_registration_number == "GJ01AB1234"
        assert res.supporting_frame_count == 3

    def test_frame_diversity_vs_preprocessing_diversity(self):
        """CRITICAL: Multiple variants of 1 frame should NOT count as multiple physical frame votes."""
        engine = MultiFrameConsensusEngine(min_confirmed_score=0.70, min_confirmed_frames=2)
        obs = [
            self._make_obs(frame_id=1, pts_ms=1000.0, text="GJ01AB1234", variant="original"),
            self._make_obs(frame_id=1, pts_ms=1000.0, text="GJ01AB1234", variant="upscaled"),
            self._make_obs(frame_id=1, pts_ms=1000.0, text="GJ01AB1234", variant="contrast"),
        ]
        res = engine.resolve_track_consensus("cam01", "TRK-001", obs)
        assert res.supporting_frame_count == 1
        assert res.recognition_status == "PROBABLE"

    def test_probable_status_single_frame(self):
        engine = MultiFrameConsensusEngine(min_confirmed_score=0.70, min_probable_score=0.50, min_confirmed_frames=2)
        obs = [self._make_obs(frame_id=1, pts_ms=1000.0, text="GJ01AB1234")]
        res = engine.resolve_track_consensus("cam01", "TRK-001", obs)
        assert res.recognition_status == "PROBABLE"
        assert res.final_registration_number == "GJ01AB1234"

    def test_uncertain_status_conflicting_readings(self):
        engine = MultiFrameConsensusEngine(min_confirmed_score=0.70, min_probable_score=0.50, min_confirmed_frames=2)
        obs = [
            self._make_obs(frame_id=1, pts_ms=1000.0, text="GJ01AB1234", score=0.40, is_valid_fmt=False),
            self._make_obs(frame_id=2, pts_ms=1080.0, text="MH12CD5678", score=0.40, is_valid_fmt=False),
        ]
        res = engine.resolve_track_consensus("cam01", "TRK-001", obs)
        assert res.recognition_status in ("UNCERTAIN", "UNREADABLE")
        assert res.final_registration_number is None

    def test_unreadable_status_empty_observations(self):
        engine = MultiFrameConsensusEngine()
        res = engine.resolve_track_consensus("cam01", "TRK-001", [])
        assert res.recognition_status == "UNREADABLE"
        assert res.final_registration_number is None

    def test_fuzzy_clustering_merges_ocr_char_confusion(self):
        engine = MultiFrameConsensusEngine(enable_fuzzy_clustering=True, max_fuzzy_distance=1)
        obs = [
            self._make_obs(frame_id=1, pts_ms=1000.0, text="GJ01AB1234", score=0.85),
            self._make_obs(frame_id=2, pts_ms=1080.0, text="GJ01A81234", score=0.80),
        ]
        res = engine.resolve_track_consensus("cam01", "TRK-001", obs)
        assert res.final_registration_number == "GJ01AB1234"
        assert res.supporting_frame_count == 2
        assert res.recognition_status == "CONFIRMED"

    def test_pts_bounds_preserved(self):
        engine = MultiFrameConsensusEngine()
        obs = [
            self._make_obs(frame_id=1, pts_ms=10000.0, text="GJ01AB1234"),
            self._make_obs(frame_id=2, pts_ms=12500.0, text="GJ01AB1234"),
        ]
        res = engine.resolve_track_consensus("cam01", "TRK-001", obs)
        assert res.first_seen_pts_ms == pytest.approx(10000.0)
        assert res.last_seen_pts_ms == pytest.approx(12500.0)
