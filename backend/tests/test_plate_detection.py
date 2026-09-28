"""Unit tests for Step 5 - Best-Frame Selection & License Plate Detection.

Tests do NOT require a live Sentinel stream or GPU.
All frames, crops, and detections are mocked with numpy arrays and synthetic data.
"""
from __future__ import annotations

import time
import numpy as np
import pytest

from src.ai.schemas import VehicleFrameCandidate, PlateCandidate
from src.ai.frame_selector import (
    compute_sharpness,
    compute_brightness,
    compute_contrast,
    compute_truncation,
    score_frame_extended,
    evaluate_frame_extended,
    select_top_vehicle_frames,
    FrameQualityWeights,
)
from src.ai.plate_quality import validate_plate_bbox, score_plate, PlateQualityMetrics
from src.ai.plate_preprocessor import (
    crop_plate_with_margin,
    generate_preprocessing_variants,
    warp_perspective,
)
from src.ai.candidate_manager import CandidateManager, _iou


# ════════════════════════════════════════════════════════════════════════════
#  1. Frame Quality & Extended Metrics
# ════════════════════════════════════════════════════════════════════════════

class TestVehicleFrameQuality:
    def _make_crop(self, h=100, w=200, fill=128) -> np.ndarray:
        crop = np.full((h, w, 3), fill, dtype=np.uint8)
        # add noise for non-zero contrast and sharpness
        crop[10:50, 10:50] = 50
        crop[50:90, 50:90] = 200
        return crop

    def test_sharpness_computation(self):
        crop = self._make_crop()
        s = compute_sharpness(crop)
        assert s > 0.0

    def test_contrast_computation(self):
        crop = self._make_crop()
        c = compute_contrast(crop)
        assert c > 0.0

    def test_truncation_full_inside(self):
        # [x1, y1, x2, y2], frame_h, frame_w
        t = compute_truncation([50, 50, 200, 200], frame_h=480, frame_w=640)
        assert t == pytest.approx(1.0)

    def test_truncation_partial_outside(self):
        # 50% of box outside right boundary
        t = compute_truncation([500, 50, 780, 200], frame_h=480, frame_w=640)
        assert 0.0 < t < 1.0

    def test_truncation_completely_outside(self):
        t = compute_truncation([700, 50, 800, 200], frame_h=480, frame_w=640)
        assert t == pytest.approx(0.0)

    def test_score_frame_extended_returns_tuple(self):
        score, ns, nb, nc, na, nt = score_frame_extended(
            sharpness=400.0, bbox_area=50000, brightness=128.0, contrast=30.0, truncation=1.0
        )
        assert 0.0 <= score <= 1.0
        assert nt == pytest.approx(1.0)

    def test_evaluate_frame_extended(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cand = evaluate_frame_extended(
            camera_id="cam01",
            track_id="TRK-001",
            vehicle_class="car",
            detection_confidence=0.85,
            frame=frame,
            bbox=[100, 100, 300, 300],
            media_pts_ms=1500.0,
            local_receive_monotonic=time.monotonic(),
            frame_sequence=12,
        )
        assert cand.camera_id == "cam01"
        assert cand.track_id == "TRK-001"
        assert cand.has_valid_pts is True
        assert cand.pts_ms == pytest.approx(1500.0)
        assert cand.vehicle_quality_score > 0.0

    def test_select_top_vehicle_frames(self):
        cands = []
        for i in range(10):
            cands.append(
                VehicleFrameCandidate(
                    camera_id="cam01", track_id="TRK-001", frame_id=i,
                    pts_ms=1000.0 + i * 40, has_valid_pts=True, local_receive_monotonic=time.monotonic(),
                    bbox=[10, 10, 100, 100], vehicle_class="car", detection_confidence=0.8,
                    sharpness_score=0.1 * i, brightness_score=0.8, contrast_score=0.8,
                    area_score=0.8, truncation_score=1.0, vehicle_quality_score=0.1 * i,
                )
            )
        top3 = select_top_vehicle_frames(cands, top_n=3)
        assert len(top3) == 3
        # Top score should be frame_id=9 (quality_score=0.9)
        assert top3[0].frame_id == 9
        assert top3[0].vehicle_quality_score == pytest.approx(0.9)


# ════════════════════════════════════════════════════════════════════════════
#  2. Bounding Box Validation
# ════════════════════════════════════════════════════════════════════════════

class TestPlateBboxValidation:
    def test_valid_plate_bbox(self):
        valid, reason = validate_plate_bbox(
            plate_bbox=[100, 100, 200, 140], frame_h=480, frame_w=640, plate_confidence=0.75
        )
        assert valid is True
        assert reason is None

    def test_out_of_bounds_bbox(self):
        valid, reason = validate_plate_bbox(
            plate_bbox=[600, 100, 680, 140], frame_h=480, frame_w=640, plate_confidence=0.75
        )
        assert valid is False
        assert "out_of_frame_bounds" in reason

    def test_low_confidence_rejection(self):
        valid, reason = validate_plate_bbox(
            plate_bbox=[100, 100, 200, 140], frame_h=480, frame_w=640,
            plate_confidence=0.20, min_confidence=0.35
        )
        assert valid is False
        assert "low_confidence" in reason

    def test_too_small_area_rejection(self):
        valid, reason = validate_plate_bbox(
            plate_bbox=[100, 100, 105, 105], frame_h=480, frame_w=640,
            plate_confidence=0.80, min_area=150
        )
        assert valid is False
        assert "too_small" in reason

    def test_negative_dimension_rejection(self):
        valid, reason = validate_plate_bbox(
            plate_bbox=[200, 100, 100, 140], frame_h=480, frame_w=640, plate_confidence=0.80
        )
        assert valid is False
        assert "non_positive" in reason


# ════════════════════════════════════════════════════════════════════════════
#  3. Plate Quality Scoring
# ════════════════════════════════════════════════════════════════════════════

class TestPlateQualityScoring:
    def test_score_plate_valid_crop(self):
        crop = np.random.randint(0, 255, (30, 90, 3), dtype=np.uint8)
        metrics = score_plate(crop)
        assert isinstance(metrics, PlateQualityMetrics)
        assert 0.0 <= metrics.composite_score <= 1.0
        assert metrics.sharpness >= 0.0

    def test_score_plate_empty_crop(self):
        crop = np.zeros((0, 0, 3), dtype=np.uint8)
        metrics = score_plate(crop)
        assert metrics.composite_score == 0.0


# ════════════════════════════════════════════════════════════════════════════
#  4. Plate Crop & Preprocessing Variants
# ════════════════════════════════════════════════════════════════════════════

class TestPlatePreprocessing:
    def test_crop_plate_with_margin(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        crop, clamped = crop_plate_with_margin(frame, [100, 100, 200, 150], margin_percent=0.10)
        assert crop.shape[0] > 50  # expanded height
        assert crop.shape[1] > 100 # expanded width
        assert clamped[0] <= 100 and clamped[1] <= 100

    def test_generate_preprocessing_variants(self):
        crop = np.random.randint(0, 255, (30, 90, 3), dtype=np.uint8)
        variants = generate_preprocessing_variants(crop, scale_factor=2.0, enable_denoise=False)

        assert "original" in variants
        assert "grayscale" in variants
        assert "upscaled" in variants
        assert "contrast" in variants
        assert "sharpened" in variants

        # Original must be untouched 3-channel
        assert variants["original"].ndim == 3
        # Grayscale variants must be 2D
        assert variants["grayscale"].ndim == 2
        # Upscaled must be 2x size
        assert variants["upscaled"].shape[1] == 180
        assert variants["upscaled"].shape[0] == 60

    def test_warp_perspective_optional(self):
        crop = np.random.randint(0, 255, (40, 120, 3), dtype=np.uint8)
        corners = np.array([[0, 0], [119, 0], [119, 39], [0, 39]], dtype=np.float32)
        warped = warp_perspective(crop, corners, target_width=100, target_height=30)
        assert warped is not None
        assert warped.shape == (30, 100, 3)

    def test_warp_perspective_none_when_no_corners(self):
        crop = np.random.randint(0, 255, (40, 120, 3), dtype=np.uint8)
        assert warp_perspective(crop, None) is None


# ════════════════════════════════════════════════════════════════════════════
#  5. Candidate Manager & Deduplication
# ════════════════════════════════════════════════════════════════════════════

class TestCandidateManager:
    def _make_candidate(
        self,
        track_id: str,
        frame_id: int,
        pts_ms: Optional[float],
        bbox: list,
        quality: float,
        conf: float = 0.70,
        rejection_reason: Optional[str] = None,
    ) -> PlateCandidate:
        return PlateCandidate(
            camera_id="cam01",
            track_id=track_id,
            frame_id=frame_id,
            pts_ms=pts_ms,
            has_valid_pts=(pts_ms is not None and pts_ms > 0),
            local_receive_monotonic=time.monotonic(),
            plate_bbox=bbox,
            plate_confidence=conf,
            detector_version="test_v1",
            plate_quality_score=quality,
            plate_width=bbox[2] - bbox[0],
            plate_height=bbox[3] - bbox[1],
            sharpness_score=quality,
            brightness_score=0.8,
            contrast_score=0.8,
            original_crop_path=None,
            processed_crop_paths={},
            rejection_reason=rejection_reason,
        )

    def test_add_and_accepted_filtering(self):
        mgr = CandidateManager()
        mgr.add_candidate(self._make_candidate("TRK-001", 1, 1000.0, [10, 10, 100, 50], quality=0.8))
        mgr.add_candidate(self._make_candidate("TRK-001", 2, 1040.0, [10, 10, 100, 50], quality=0.4, rejection_reason="too_small"))

        assert len(mgr.get_all_accepted_candidates()) == 1
        assert len(mgr.get_all_rejected_candidates()) == 1

    def test_deduplication_removes_near_identical_candidates(self):
        mgr = CandidateManager(dedup_iou_threshold=0.50, dedup_frame_window=10)
        # Frame 1 and Frame 3 are close in time and overlap heavily -> lower quality one should be dropped
        cand1 = self._make_candidate("TRK-001", 1, 1000.0, [100, 100, 200, 140], quality=0.60)
        cand2 = self._make_candidate("TRK-001", 3, 1080.0, [102, 101, 202, 141], quality=0.90) # Higher quality

        mgr.add_candidate(cand1)
        mgr.add_candidate(cand2)

        deduped = mgr.deduplicate_track_candidates("TRK-001")
        assert len(deduped) == 1
        # Retained candidate must be cand2 (higher quality/combined_score)
        assert deduped[0].frame_id == 3
        assert deduped[0].plate_quality_score == pytest.approx(0.90)

    def test_deduplication_retains_spatially_distinct_candidates(self):
        mgr = CandidateManager(dedup_iou_threshold=0.50, dedup_frame_window=10)
        cand1 = self._make_candidate("TRK-001", 1, 1000.0, [100, 100, 200, 140], quality=0.60)
        cand2 = self._make_candidate("TRK-001", 3, 1080.0, [400, 400, 500, 440], quality=0.90) # Distinct location

        mgr.add_candidate(cand1)
        mgr.add_candidate(cand2)

        deduped = mgr.deduplicate_track_candidates("TRK-001")
        assert len(deduped) == 2

    def test_top_n_limit_per_track(self):
        mgr = CandidateManager(best_plates_per_track=2)
        for i in range(5):
            # Non-overlapping locations so dedup doesn't drop them
            mgr.add_candidate(
                self._make_candidate("TRK-001", i * 20, 1000.0 + i * 800, [i * 100, 10, i * 100 + 80, 50], quality=0.1 * (i + 1))
            )
        top = mgr.get_top_candidates_per_track("TRK-001")
        assert len(top) == 2
        # Highest scores are i=4 (quality=0.5) and i=3 (quality=0.4)
        assert top[0].frame_id == 80
        assert top[1].frame_id == 60


# ════════════════════════════════════════════════════════════════════════════
#  6. PTS & Timing Rules
# ════════════════════════════════════════════════════════════════════════════

class TestPTSHandling:
    def test_valid_pts_propagation(self):
        cand = PlateCandidate(
            camera_id="cam01", track_id="TRK-001", frame_id=42, pts_ms=183420.0,
            has_valid_pts=True, local_receive_monotonic=100.5, plate_bbox=[10, 10, 100, 40],
            plate_confidence=0.8, detector_version="test", plate_quality_score=0.7,
            plate_width=90, plate_height=30, sharpness_score=0.7, brightness_score=0.7, contrast_score=0.7
        )
        cdict = cand.to_dict()
        assert cdict["pts_ms"] == pytest.approx(183420.0)
        assert cdict["has_valid_pts"] is True

    def test_missing_pts_explicitly_flagged(self):
        cand = PlateCandidate(
            camera_id="cam01", track_id="TRK-001", frame_id=42, pts_ms=None,
            has_valid_pts=False, local_receive_monotonic=100.5, plate_bbox=[10, 10, 100, 40],
            plate_confidence=0.8, detector_version="test", plate_quality_score=0.7,
            plate_width=90, plate_height=30, sharpness_score=0.7, brightness_score=0.7, contrast_score=0.7
        )
        cdict = cand.to_dict()
        assert cdict["pts_ms"] is None
        assert cdict["has_valid_pts"] is False

    def test_no_registration_number_field_in_schema(self):
        """STEP 5 BOUNDARY TEST: Ensure registration_number is not present in output."""
        cand = PlateCandidate(
            camera_id="cam01", track_id="TRK-001", frame_id=42, pts_ms=1000.0,
            has_valid_pts=True, local_receive_monotonic=100.5, plate_bbox=[10, 10, 100, 40],
            plate_confidence=0.8, detector_version="test", plate_quality_score=0.7,
            plate_width=90, plate_height=30, sharpness_score=0.7, brightness_score=0.7, contrast_score=0.7
        )
        cdict = cand.to_dict()
        assert "registration_number" not in cdict
        assert "text" not in cdict
        assert "ocr_text" not in cdict
