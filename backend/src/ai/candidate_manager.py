"""Candidate manager for multi-frame plate candidates and cross-frame deduplication.

CROSS-FRAME DEDUPLICATION & SELECTION:
  1. Multi-frame candidate pool per vehicle track (a track may yield 5-10 plate candidates).
  2. Deduplication based on:
     - Same track ID
     - Plate bounding-box IoU overlap (> threshold)
     - Spatial-temporal frame proximity (< frame_window)
     When duplicate candidates occur, retain the candidate with highest combined_score.
  3. Top-N selection per vehicle track (configurable BEST_PLATE_CANDIDATES=5).
  4. Export to JSONL (plate_candidates.jsonl) and JSON summary (plate_summary.json).

STEP 6 INPUT FORMAT:
  Produces clean structured JSON/JSONL output consumed directly by Step 6 (ANPR/OCR).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.ai.schemas import PlateCandidate
from src.common.logging import get_logger

logger = get_logger("candidate_manager")


def _iou(boxA: List[int], boxB: List[int]) -> float:
    """Intersection-over-Union between two [x1, y1, x2, y2] bboxes."""
    if not boxA or not boxB or len(boxA) != 4 or len(boxB) != 4:
        return 0.0
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
    return inter / float(union) if union > 0 else 0.0


class CandidateManager:
    """Manages collection, deduplication, ranking, and export of PlateCandidate objects."""

    def __init__(
        self,
        best_plates_per_track: int = 5,
        dedup_iou_threshold: float = 0.50,
        dedup_frame_window: int = 10,
    ):
        self.best_plates_per_track = best_plates_per_track
        self.dedup_iou_threshold = dedup_iou_threshold
        self.dedup_frame_window = dedup_frame_window

        # Storage: track_id -> List[PlateCandidate] (includes accepted and rejected for metrics)
        self._candidates_by_track: Dict[str, List[PlateCandidate]] = defaultdict(list)
        self._total_accepted = 0
        self._total_rejected = 0

    def add_candidate(self, candidate: PlateCandidate) -> None:
        """Add a plate candidate (accepted or rejected) to the pool."""
        self._candidates_by_track[candidate.track_id].append(candidate)
        if candidate.is_accepted:
            self._total_accepted += 1
        else:
            self._total_rejected += 1

    def deduplicate_track_candidates(self, track_id: str) -> List[PlateCandidate]:
        """Deduplicate accepted candidates for a specific track.

        Keeps the highest combined_score candidate among overlapping/near-identical detections.
        """
        all_cands = self._candidates_by_track.get(track_id, [])
        accepted = [c for c in all_cands if c.is_accepted]

        if len(accepted) <= 1:
            return accepted

        # Sort by combined_score descending
        sorted_cands = sorted(accepted, key=lambda c: c.combined_score, reverse=True)
        retained: List[PlateCandidate] = []

        for cand in sorted_cands:
            is_dup = False
            for kept in retained:
                # Check spatial overlap and temporal proximity
                frame_dist = abs(cand.frame_id - kept.frame_id)
                overlap = _iou(cand.plate_bbox, kept.plate_bbox)

                if frame_dist <= self.dedup_frame_window and overlap >= self.dedup_iou_threshold:
                    is_dup = True
                    break

            if not is_dup:
                retained.append(cand)

        return retained

    def get_top_candidates_per_track(
        self, track_id: str, limit: Optional[int] = None
    ) -> List[PlateCandidate]:
        """Get deduplicated top-N accepted candidates for a track, sorted by combined score."""
        n = limit or self.best_plates_per_track
        deduped = self.deduplicate_track_candidates(track_id)
        deduped_sorted = sorted(deduped, key=lambda c: c.combined_score, reverse=True)
        return deduped_sorted[:n]

    def get_all_accepted_candidates(self) -> List[PlateCandidate]:
        """Return all deduplicated top-N accepted candidates across all tracks."""
        result = []
        for track_id in self._candidates_by_track.keys():
            result.extend(self.get_top_candidates_per_track(track_id))
        return result

    def get_all_rejected_candidates(self) -> List[PlateCandidate]:
        """Return all rejected candidates across all tracks (for rejection diagnostics)."""
        rejected = []
        for cands in self._candidates_by_track.values():
            rejected.extend([c for c in cands if not c.is_accepted])
        return rejected

    def export_jsonl(self, output_path: str) -> int:
        """Write all deduplicated top-N accepted candidates to JSONL.

        Returns number of records written.
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        accepted_cands = self.get_all_accepted_candidates()

        with open(out_file, "w", encoding="utf-8") as f:
            for cand in accepted_cands:
                f.write(json.dumps(cand.to_dict()) + "\n")

        logger.info("Exported %d plate candidates to JSONL: %s", len(accepted_cands), output_path)
        return len(accepted_cands)

    def export_summary_json(self, output_path: str, camera_id: str) -> Dict[str, Any]:
        """Write per-track plate summary JSON for Step 6 consumption.

        Structure:
        {
          "camera_id": "cam01",
          "total_tracks": 10,
          "tracks_with_plates": 8,
          "total_plate_candidates": 24,
          "tracks": [
             {
               "track_id": "TRK-001",
               "plate_candidate_count": 3,
               "best_plate_candidate": { ... },
               "plate_candidates": [ ... ]
             }
          ]
        }
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        track_summaries = []
        tracks_with_plates = 0
        total_retained_candidates = 0

        for track_id in sorted(self._candidates_by_track.keys()):
            cands = self.get_top_candidates_per_track(track_id)
            cand_dicts = [c.to_dict() for c in cands]

            if cands:
                tracks_with_plates += 1
                total_retained_candidates += len(cands)
                best_cand = cand_dicts[0]
            else:
                best_cand = None

            track_summaries.append({
                "track_id": track_id,
                "camera_id": camera_id,
                "plate_candidate_count": len(cands),
                "best_plate_candidate": best_cand,
                "plate_candidates": cand_dicts,
            })

        summary_payload = {
            "camera_id": camera_id,
            "total_tracks_examined": len(self._candidates_by_track),
            "tracks_with_plates": tracks_with_plates,
            "total_plate_candidates": total_retained_candidates,
            "total_raw_accepted": self._total_accepted,
            "total_raw_rejected": self._total_rejected,
            "tracks": track_summaries,
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, ensure_ascii=False)

        logger.info("Exported plate summary JSON: %s (%d tracks)", output_path, len(track_summaries))
        return summary_payload
