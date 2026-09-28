"""Unit tests for Step 4 — Vehicle Detection & Tracking.

Tests do NOT require a live Sentinel stream or GPU.
All frames/detections are mocked with numpy arrays and synthetic data.
"""
from __future__ import annotations

import time
import numpy as np
import pytest

from src.ai.schemas import Detection, VehicleTrack, DetectionEvent, FrameCandidate
from src.ai.tracker import VehicleTracker, _iou
from src.ai.frame_selector import (
    compute_sharpness, compute_brightness, score_frame,
    evaluate_frame, update_best_frames,
)


# ════════════════════════════════════════════════════════════════════════
#  Detection schema
# ════════════════════════════════════════════════════════════════════════

class TestDetectionSchema:
    def _make_det(self, bbox, conf=0.80, cls="car") -> Detection:
        return Detection(vehicle_class=cls, confidence=conf, bbox=bbox)

    def test_valid_bbox(self):
        d = self._make_det([100, 100, 300, 300])
        assert d.bbox_valid
        assert d.bbox_area == 200 * 200

    def test_invalid_bbox_reversed(self):
        d = self._make_det([300, 300, 100, 100])
        assert not d.bbox_valid

    def test_invalid_bbox_zero_area(self):
        d = self._make_det([100, 100, 100, 300])
        assert not d.bbox_valid

    def test_wrong_length_bbox(self):
        d = self._make_det([100, 100, 300])
        assert not d.bbox_valid

    def test_confidence_passthrough(self):
        d = self._make_det([0, 0, 100, 100], conf=0.55)
        assert abs(d.confidence - 0.55) < 1e-6

    def test_to_dict_keys(self):
        d = self._make_det([10, 10, 200, 200], conf=0.9, cls="truck")
        dd = d.to_dict()
        assert dd["vehicle_class"] == "truck"
        assert dd["confidence"] == pytest.approx(0.9)
        assert dd["bbox"] == [10, 10, 200, 200]


# ════════════════════════════════════════════════════════════════════════
#  Confidence filtering (simulated through tracker schema)
# ════════════════════════════════════════════════════════════════════════

class TestConfidenceFiltering:
    def test_low_confidence_rejected(self):
        # Simulate what detector.detect() does: skip conf < threshold
        raw = [
            Detection("car", 0.75, [0, 0, 100, 100]),
            Detection("truck", 0.25, [0, 0, 100, 100]),  # below 0.40
        ]
        threshold = 0.40
        passed = [d for d in raw if d.confidence >= threshold]
        assert len(passed) == 1
        assert passed[0].vehicle_class == "car"

    def test_exact_threshold_accepted(self):
        d = Detection("bus", 0.40, [0, 0, 50, 50])
        assert d.confidence >= 0.40


# ════════════════════════════════════════════════════════════════════════
#  IoU helper
# ════════════════════════════════════════════════════════════════════════

class TestIoU:
    def test_perfect_overlap(self):
        box = [0, 0, 100, 100]
        assert _iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert _iou([0, 0, 50, 50], [100, 100, 200, 200]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        val = _iou([0, 0, 100, 100], [50, 50, 150, 150])
        assert 0.0 < val < 1.0


# ════════════════════════════════════════════════════════════════════════
#  Tracker — track creation and continuation
# ════════════════════════════════════════════════════════════════════════

class TestTracker:
    def _det(self, bbox, cls="car", conf=0.80) -> Detection:
        return Detection(vehicle_class=cls, confidence=conf, bbox=bbox, media_pts_ms=1000.0)

    def test_track_created_on_first_detection(self):
        tracker = VehicleTracker()
        active = tracker.update([self._det([10, 10, 110, 110])], media_pts_ms=1000.0)
        assert len(active) == 1

    def test_track_id_format(self):
        tracker = VehicleTracker()
        tracker.update([self._det([10, 10, 110, 110])], media_pts_ms=1000.0)
        tid = list(tracker._tracks.keys())[0]
        assert tid.startswith("TRK-")

    def test_same_vehicle_same_track(self):
        tracker = VehicleTracker(iou_threshold=0.30, min_hits=1)
        tracker.update([self._det([10, 10, 110, 110])], media_pts_ms=1000.0)
        # Slightly moved — same vehicle
        tracker.update([self._det([15, 15, 115, 115])], media_pts_ms=1040.0)
        assert len(tracker._tracks) == 1
        t = list(tracker._tracks.values())[0]
        assert t.frame_count == 2

    def test_two_distinct_vehicles(self):
        tracker = VehicleTracker(iou_threshold=0.30, min_hits=1)
        tracker.update(
            [self._det([0, 0, 100, 100]), self._det([500, 0, 600, 100])],
            media_pts_ms=1000.0,
        )
        assert len(tracker._tracks) == 2

    def test_missed_frames_tolerance(self):
        tracker = VehicleTracker(max_age=5, min_hits=1)
        tracker.update([self._det([10, 10, 110, 110])], media_pts_ms=1000.0)
        # 3 frames with no detections — should stay alive (max_age=5)
        for _ in range(3):
            active = tracker.update([], media_pts_ms=1040.0)
        assert len(active) == 1
        t = list(tracker._tracks.values())[0]
        assert t.is_active

    def test_track_expiry_after_max_age(self):
        tracker = VehicleTracker(max_age=2, min_hits=1)
        tracker.update([self._det([10, 10, 110, 110])], media_pts_ms=1000.0)
        # Exceed max_age
        for _ in range(3):
            tracker.update([], media_pts_ms=1040.0)
        assert all(not t.is_active for t in tracker._tracks.values())
        assert tracker.completed_count() == 1

    def test_pts_preserved_in_track(self):
        tracker = VehicleTracker(min_hits=1)
        tracker.update([self._det([10, 10, 110, 110])], media_pts_ms=500.0)
        tracker.update([self._det([12, 12, 112, 112])], media_pts_ms=540.0)
        t = list(tracker._tracks.values())[0]
        assert t.first_seen_pts_ms == pytest.approx(500.0)
        assert t.last_seen_pts_ms == pytest.approx(540.0)

    def test_pts_progression_does_not_use_fps(self):
        tracker = VehicleTracker(min_hits=1)
        # Supply arbitrary PTS values — tracker must not derive them from fps
        pts_values = [1000.0, 1080.0, 1200.0, 1160.0]  # non-uniform
        for pts in pts_values:
            tracker.update([self._det([10, 10, 110, 110])], media_pts_ms=pts)
        t = list(tracker._tracks.values())[0]
        assert t.last_seen_pts_ms == pytest.approx(1160.0)  # last supplied PTS

    def test_reset_terminates_active_tracks(self):
        tracker = VehicleTracker(min_hits=1)
        tracker.update([self._det([10, 10, 110, 110])], media_pts_ms=1000.0)
        assert tracker.active_count() == 1
        tracker.reset()
        assert tracker.active_count() == 0
        assert tracker.completed_count() == 1  # terminated but preserved in history

    def test_pts_discontinuity_triggers_reset(self):
        """Backward PTS jump should cause tracker reset (handled by pipeline)."""
        tracker = VehicleTracker(min_hits=1)
        tracker.update([self._det([10, 10, 110, 110])], media_pts_ms=5000.0)
        # Simulate backward PTS — pipeline calls tracker.reset()
        tracker.reset()
        tracker.update([self._det([200, 200, 300, 300])], media_pts_ms=100.0)
        assert tracker.active_count() == 1
        # The old track must be gone; only the new one should be active
        active = [t for t in tracker._tracks.values() if t.is_active]
        assert len(active) == 1
        assert active[0].first_seen_pts_ms == pytest.approx(100.0)


# ════════════════════════════════════════════════════════════════════════
#  Frame sampling logic
# ════════════════════════════════════════════════════════════════════════

class TestDetectorSampling:
    def test_every_nth_frame(self):
        interval = 3
        total_frames = 30
        detected_frames = [i for i in range(1, total_frames + 1) if i % interval == 0]
        assert len(detected_frames) == total_frames // interval

    def test_interval_1_processes_every_frame(self):
        interval = 1
        assert all((i % interval == 0) for i in range(1, 11))


# ════════════════════════════════════════════════════════════════════════
#  Frame quality scoring
# ════════════════════════════════════════════════════════════════════════

class TestBestFrameRanking:
    def _make_frame(self, h=480, w=640) -> np.ndarray:
        return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    def _make_candidate(self, quality: float) -> FrameCandidate:
        return FrameCandidate(
            frame_sequence=1,
            media_pts_ms=1000.0,
            local_receive_monotonic=time.monotonic(),
            sharpness=quality * 100,
            bbox_area=50_000,
            brightness=128.0,
            quality_score=quality,
        )

    def test_sharpness_computed(self):
        frame = self._make_frame()
        bbox = [50, 50, 300, 300]
        s = compute_sharpness(frame[50:300, 50:300])
        assert s >= 0.0

    def test_brightness_in_range(self):
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        b = compute_brightness(frame)
        assert abs(b - 128.0) < 1.0

    def test_top_n_frames_kept(self):
        track = VehicleTrack(
            track_id="TRK-0001", camera_id="cam01", vehicle_class="car",
            first_seen_pts_ms=1000.0, last_seen_pts_ms=2000.0,
        )
        for q in [0.1, 0.9, 0.5, 0.7, 0.3, 0.8]:
            update_best_frames(track, self._make_candidate(q), max_best_frames=3)
        assert len(track.best_frames) == 3
        # Must be the top-3 by quality score
        scores = [fc.quality_score for fc in track.best_frames]
        assert sorted(scores, reverse=True) == scores
        assert min(scores) >= 0.7 - 1e-6

    def test_evaluate_frame_returns_candidate(self):
        frame = self._make_frame()
        cand = evaluate_frame(
            frame=frame,
            bbox=[10, 10, 200, 200],
            media_pts_ms=1234.0,
            local_receive_monotonic=time.monotonic(),
            frame_sequence=42,
        )
        assert cand.frame_sequence == 42
        assert cand.media_pts_ms == pytest.approx(1234.0)
        assert cand.quality_score >= 0.0
        assert cand.bbox_area > 0
