"""Sequence and Ordering Engine for Step 11 Vehicle Journey Reconstruction.

Applies:
  MODE A: SOURCE_TIME ordering when common calendar clock is RESOLVED.
  MODE B: CAMERA_LOCAL grouping when common calendar clock is NOT_RESOLVED.
"""
from __future__ import annotations

from typing import List, Tuple

from src.journey.models import CameraObservationSegment, OrderingMode
from src.journey.temporal import can_order_globally, parse_source_time


class SequenceBuilder:
    """Orders camera observation segments into a sequence according to timing certainty."""

    def build_sequence(
        self,
        segments: List[CameraObservationSegment],
    ) -> Tuple[List[CameraObservationSegment], OrderingMode]:
        """Order segments based on timing validation.

        Returns:
            Tuple of (ordered_segments, ordering_mode).
        """
        if not segments:
            return [], OrderingMode.CAMERA_LOCAL

        if can_order_globally(segments):
            # MODE A: All segments have resolved source_time -> sort chronologically
            sorted_segments = sorted(
                segments,
                key=lambda s: parse_source_time(s.source_time),  # type: ignore
            )
            return sorted_segments, OrderingMode.SOURCE_TIME

        # MODE B: Unresolved source_time -> group by camera, sort within each camera by camera-local PTS
        by_camera = {}
        for s in segments:
            by_camera.setdefault(s.camera_id, []).append(s)

        ordered: List[CameraObservationSegment] = []
        for cam_id in sorted(by_camera.keys()):
            cam_segs = sorted(by_camera[cam_id], key=lambda s: s.first_seen_pts_ms or 0.0)
            ordered.extend(cam_segs)

        return ordered, OrderingMode.CAMERA_LOCAL
