"""IoU-based multi-object tracker for vehicle tracks.

Design rationale:
  - Simple, dependency-free centroid/IoU tracker.
  - Modular: can be replaced by ByteTrack / BotSORT / DeepSORT in Step 5
    by swapping out this module while keeping the VehicleTrack schema unchanged.
  - PTS from the video container is used for temporal state; FPS is never used.
  - Stream disconnect / PTS discontinuity signals a full track reset.

Track lifecycle:
  TENTATIVE  →  (confirmed after min_hits detections)  →  CONFIRMED
  CONFIRMED  →  (missed_frames > max_age)              →  EXPIRED (removed)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from src.ai.schemas import Detection, VehicleTrack, FrameCandidate
from src.common.logging import get_logger

logger = get_logger("vehicle_tracker")


def _iou(boxA: List[int], boxB: List[int]) -> float:
    """Compute Intersection-over-Union between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


class VehicleTracker:
    """IoU-based multi-object tracker.

    Replaceability contract:
      update(detections, media_pts_ms) → List[VehicleTrack]
      reset()                          → clears all state (on stream disconnect)
    """

    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_age: int = 30,           # frames of consecutive misses before expiry
        min_hits: int = 2,           # detections needed before track is confirmed
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits

        self._tracks: Dict[str, VehicleTrack] = {}
        self._next_id: int = 1

    def _new_track_id(self) -> str:
        tid = f"TRK-{self._next_id:04d}"
        self._next_id += 1
        return tid

    def _match(
        self, detections: List[Detection]
    ) -> Tuple[Dict[str, Detection], List[Detection], List[str]]:
        """
        Greedy IoU matching between current active tracks and new detections.

        Returns:
          matched   : {track_id → Detection}
          unmatched_dets : detections that did not match any track
          unmatched_trks : track_ids that had no matching detection
        """
        active_ids = [tid for tid, t in self._tracks.items() if t.is_active]

        if not active_ids or not detections:
            return {}, list(detections), active_ids

        # Build IoU matrix [n_tracks x n_detections]
        iou_mat = np.zeros((len(active_ids), len(detections)), dtype=np.float32)
        for ti, tid in enumerate(active_ids):
            for di, det in enumerate(detections):
                iou_mat[ti, di] = _iou(self._tracks[tid].current_bbox, det.bbox)

        matched: Dict[str, Detection] = {}
        used_dets = set()
        used_trks = set()

        # Greedy: pick highest IoU pairs above threshold
        flat = np.argsort(iou_mat.ravel())[::-1]
        for idx in flat:
            ti = idx // len(detections)
            di = idx % len(detections)
            if iou_mat[ti, di] < self.iou_threshold:
                break
            if ti in used_trks or di in used_dets:
                continue
            matched[active_ids[ti]] = detections[di]
            used_trks.add(ti)
            used_dets.add(di)

        unmatched_dets = [d for i, d in enumerate(detections) if i not in used_dets]
        unmatched_trks = [active_ids[ti] for ti in range(len(active_ids)) if ti not in used_trks]
        return matched, unmatched_dets, unmatched_trks

    def update(
        self,
        detections: List[Detection],
        media_pts_ms: Optional[float],
    ) -> List[VehicleTrack]:
        """Process one frame's detections and return all currently active tracks.

        Args:
            detections: Vehicle detections for the current frame.
            media_pts_ms: Container PTS for the current frame.

        Returns:
            List of VehicleTrack objects currently active (including tentative).
        """
        matched, unmatched_dets, unmatched_trks = self._match(detections)

        # 1. Update matched tracks
        for tid, det in matched.items():
            t = self._tracks[tid]
            t.last_seen_pts_ms = media_pts_ms
            t.current_bbox = det.bbox
            t.frame_count += 1
            t.missed_frames = 0
            t.confidences.append(det.confidence)

        # 2. Increment missed counter for unmatched tracks
        expired_ids = []
        for tid in unmatched_trks:
            t = self._tracks[tid]
            t.missed_frames += 1
            if t.missed_frames > self.max_age:
                t.is_active = False
                expired_ids.append(tid)
                logger.debug("Track %s expired after %d missed frames.", tid, t.missed_frames)

        # 3. Create new tentative tracks for unmatched detections
        for det in unmatched_dets:
            tid = self._new_track_id()
            self._tracks[tid] = VehicleTrack(
                track_id=tid,
                camera_id="",   # set by pipeline
                vehicle_class=det.vehicle_class,
                first_seen_pts_ms=media_pts_ms,
                last_seen_pts_ms=media_pts_ms,
                frame_count=1,
                missed_frames=0,
                current_bbox=det.bbox,
                confidences=[det.confidence],
            )

        return [t for t in self._tracks.values() if t.is_active]

    def confirmed_tracks(self) -> List[VehicleTrack]:
        """Return only tracks that have been seen at least min_hits times."""
        return [
            t for t in self._tracks.values()
            if t.is_active and t.frame_count >= self.min_hits
        ]

    def all_tracks(self) -> List[VehicleTrack]:
        """Return all tracks including expired ones (for summary export)."""
        return list(self._tracks.values())

    def active_count(self) -> int:
        return sum(1 for t in self._tracks.values() if t.is_active)

    def completed_count(self) -> int:
        return sum(1 for t in self._tracks.values() if not t.is_active)

    def reset(self) -> None:
        """Terminate all active tracks — called on stream reconnect or PTS discontinuity.

        Tracks are preserved in the history (for later export) but marked inactive.
        """
        count = 0
        for t in self._tracks.values():
            if t.is_active:
                t.is_active = False
                count += 1
        if count:
            logger.warning(
                "Tracker reset: %d active track(s) terminated due to stream discontinuity.", count
            )
