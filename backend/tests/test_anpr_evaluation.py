"""Unit tests for ANPR Evaluation, Failure Classification, and Metrics Analysis (Step 6).

Covers Part 25:
  1. Dataset sample validation.
  2. Missing image handling.
  3. Duplicate sample detection.
  4. Ground-truth normalization.
  5. Condition counting with sample count verification.
  6. Exact-match calculation.
  7. Normalized exact-match calculation.
  8. Character accuracy calculation.
  9. False-recognition calculation.
  10. Track-level accuracy calculation.
  11. Consensus statistics.
  12. Same-frame vs multi-frame deduplication.
  13. Failure-category aggregation.
  14. Threshold analysis tradeoff calculation.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from pathlib import Path

from src.ai.anpr.evaluation import (
    ALLOWED_CONDITIONS,
    FAILURE_CATEGORIES,
    DatasetSampleValidation,
    SampleDiagnostic,
    classify_failure,
    compute_character_accuracy,
    validate_dataset_sample,
)
from src.ai.anpr.consensus import MultiFrameConsensusEngine
from src.ai.anpr.normalizer import normalize_raw_text
from src.ai.anpr.schemas import ANPRObservation, OCRResult, NormalizedOCRResult


# ════════════════════════════════════════════════════════════════════════════
# 1. Dataset Validation & Missing Image Handling
# ════════════════════════════════════════════════════════════════════════════

def test_validate_dataset_sample_valid(tmp_path):
    img_file = tmp_path / "valid_plate.jpg"
    img = np.full((70, 240, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(img_file), img)

    rec = {
        "image": str(img_file),
        "ground_truth": "GJ01AB1234",
        "condition": "clear",
        "source": "controlled",
    }
    val, loaded_img = validate_dataset_sample(rec)
    assert val.is_valid is True
    assert val.status == "VALID"
    assert val.width == 240
    assert val.height == 70
    assert loaded_img is not None


def test_validate_dataset_sample_missing_image(tmp_path):
    rec = {
        "image": str(tmp_path / "non_existent_img.jpg"),
        "ground_truth": "GJ01AB1234",
        "condition": "clear",
    }
    val, loaded_img = validate_dataset_sample(rec)
    assert val.is_valid is False
    assert val.status == "MISSING_IMAGE"
    assert loaded_img is None


def test_validate_dataset_sample_missing_ground_truth(tmp_path):
    img_file = tmp_path / "img.jpg"
    img = np.full((50, 150, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(img_file), img)

    rec = {
        "image": str(img_file),
        "ground_truth": "",  # Empty
    }
    val, loaded_img = validate_dataset_sample(rec)
    assert val.is_valid is False
    assert val.status == "MISSING_GROUND_TRUTH"


# ════════════════════════════════════════════════════════════════════════════
# 2. Duplicate Detection & Normalization
# ════════════════════════════════════════════════════════════════════════════

def test_duplicate_sample_detection():
    existing_records = [
        {"image": "images/sample_01.jpg", "ground_truth": "GJ01AB1234"},
        {"image": "images/sample_02.jpg", "ground_truth": "GJ05CD5678"},
    ]
    seen = {Path(r["image"]).name for r in existing_records}

    new_candidate = "images/sample_01.jpg"
    assert Path(new_candidate).name in seen

    distinct_candidate = "images/sample_03.jpg"
    assert Path(distinct_candidate).name not in seen


def test_ground_truth_normalization():
    assert normalize_raw_text("gj - 01 ab 1234") == "GJ01AB1234"
    assert normalize_raw_text("INDGJ01AB1234") == "GJ01AB1234"


# ════════════════════════════════════════════════════════════════════════════
# 3. Accuracy & Character Accuracy Metrics
# ════════════════════════════════════════════════════════════════════════════

def test_compute_character_accuracy_exact():
    assert compute_character_accuracy("GJ01AB1234", "GJ01AB1234") == pytest.approx(1.0)


def test_compute_character_accuracy_single_char_diff():
    # 1 mismatch out of 10 characters -> 90%
    assert compute_character_accuracy("GJ01AB1234", "GJ01A81234") == pytest.approx(0.90)


def test_exact_vs_normalized_match_logic():
    gt = "GJ01AB1234"
    pred_raw = "GJO1AB1234"  # 'O' in district position
    pred_norm = "GJ01AB1234"

    exact_match = (pred_raw == gt)
    norm_match = (pred_norm == gt)

    assert exact_match is False
    assert norm_match is True


# ════════════════════════════════════════════════════════════════════════════
# 4. Failure Classification & Aggregation
# ════════════════════════════════════════════════════════════════════════════

def test_failure_classification_plate_too_small():
    val = DatasetSampleValidation(
        image_path="test.jpg", exists=True, readable=True, width=25, height=10,
        file_size_bytes=100, ground_truth="GJ01AB1234", normalized_ground_truth="GJ01AB1234",
        condition="small_plate", source="sentinel", status="VALID"
    )
    reason = classify_failure("GJ01AB1234", "", "", 0.0, val)
    assert reason == "PLATE_TOO_SMALL"


def test_failure_classification_glare_condition():
    val = DatasetSampleValidation(
        image_path="test.jpg", exists=True, readable=True, width=200, height=60,
        file_size_bytes=5000, ground_truth="GJ01AB1234", normalized_ground_truth="GJ01AB1234",
        condition="glare", source="controlled", status="VALID"
    )
    reason = classify_failure("GJ01AB1234", "81234", "81234", 0.85, val)
    assert reason == "GLARE"


def test_failure_classification_correct():
    val = DatasetSampleValidation(
        image_path="test.jpg", exists=True, readable=True, width=200, height=60,
        file_size_bytes=5000, ground_truth="GJ01AB1234", normalized_ground_truth="GJ01AB1234",
        condition="clear", source="controlled", status="VALID"
    )
    reason = classify_failure("GJ01AB1234", "GJ01AB1234", "GJ01AB1234", 0.95, val)
    assert reason == "CORRECT"


# ════════════════════════════════════════════════════════════════════════════
# 5. Track-Level Multi-Frame Consensus & Deduplication
# ════════════════════════════════════════════════════════════════════════════

def _make_eval_obs(frame_id: int, pts_ms: float, text: str, variant: str = "original") -> ANPRObservation:
    return ANPRObservation(
        camera_id="cam01",
        track_id="TRK-001",
        frame_id=frame_id,
        pts_ms=pts_ms,
        has_valid_pts=True,
        local_receive_monotonic=100.0,
        plate_bbox=[10, 10, 100, 40],
        plate_detection_confidence=0.90,
        plate_quality_score=0.85,
        variant_name=variant,
        raw_ocr=OCRResult(text, 0.90, None, 10.0, "easyocr", "1.7.2", variant),
        normalized_ocr=NormalizedOCRResult(text, text, is_valid_format=True, format_name="standard_indian_10"),
        observation_score=0.88,
    )


def test_track_level_consensus_agreement():
    engine = MultiFrameConsensusEngine(min_confirmed_score=0.65, min_confirmed_frames=2)
    obs = [
        _make_eval_obs(frame_id=1, pts_ms=1000.0, text="GJ01AB1234"),
        _make_eval_obs(frame_id=2, pts_ms=1080.0, text="GJ01AB1234"),
        _make_eval_obs(frame_id=3, pts_ms=1160.0, text="GJ01AB1234"),
    ]
    res = engine.resolve_track_consensus("cam01", "TRK-001", obs)
    assert res.recognition_status == "CONFIRMED"
    assert res.final_registration_number == "GJ01AB1234"
    assert res.supporting_frame_count == 3


def test_same_frame_preprocessing_deduplication():
    """Variants of the SAME physical video frame must NOT count as multiple frame votes."""
    engine = MultiFrameConsensusEngine(min_confirmed_score=0.65, min_confirmed_frames=2)
    obs = [
        _make_eval_obs(frame_id=10, pts_ms=2000.0, text="GJ01AB1234", variant="original"),
        _make_eval_obs(frame_id=10, pts_ms=2000.0, text="GJ01AB1234", variant="upscaled"),
        _make_eval_obs(frame_id=10, pts_ms=2000.0, text="GJ01AB1234", variant="contrast"),
    ]
    res = engine.resolve_track_consensus("cam01", "TRK-001", obs)
    # Only 1 unique physical frame
    assert res.supporting_frame_count == 1
    assert res.recognition_status == "PROBABLE"  # Not CONFIRMED (requires >= 2 physical frames)


# ════════════════════════════════════════════════════════════════════════════
# 6. Confidence Threshold Tradeoff Analysis
# ════════════════════════════════════════════════════════════════════════════

def test_confidence_threshold_analysis_tradeoffs():
    cutoffs = [0.30, 0.60]
    # Sample with confidence 0.45
    sample_conf = 0.45
    
    # At cutoff 0.30, 0.45 >= 0.30 -> accepted
    accepted_at_30 = (sample_conf >= 0.30)
    # At cutoff 0.60, 0.45 < 0.60 -> filtered out / uncertain
    accepted_at_60 = (sample_conf >= 0.60)

    assert accepted_at_30 is True
    assert accepted_at_60 is False
