"""Observation Correlation and Intra-Camera Grouping for Step 11.

Rules:
  1. Aggregates multiple detections within the same camera track into a single CameraObservationSegment.
  2. Keeps observations across different cameras strictly distinct (cam01 != cam07).
  3. Preserves distinct segments for multiple separate tracks on the same camera.
  4. Retains repeated camera visits (e.g. cam01 -> cam07 -> cam01).
"""
from __future__ import annotations

import collections
from typing import Dict, List, Tuple

from src.journey.models import CameraObservationSegment, CrossCameraObservation


class ObservationCorrelator:
    """Correlates individual frame observations into camera-track segments."""

    def build_camera_segments(
        self,
        observations: List[CrossCameraObservation],
    ) -> List[CameraObservationSegment]:
        """Group observations into atomic camera-track segments.

        Groups by (camera_id, track_id).
        """
        if not observations:
            return []

        # Preserves encounter order
        grouped: Dict[Tuple[str, str], List[CrossCameraObservation]] = collections.defaultdict(list)
        for obs in observations:
            key = (obs.camera_id, obs.track_id)
            grouped[key].append(obs)

        segments: List[CameraObservationSegment] = []
        seg_idx = 1

        for (cam_id, trk_id), obs_list in grouped.items():
            first_obs = obs_list[0]

            first_pts_candidates = [o.first_seen_pts_ms for o in obs_list if o.first_seen_pts_ms is not None]
            last_pts_candidates = [o.last_seen_pts_ms for o in obs_list if o.last_seen_pts_ms is not None]
            rec_pts_candidates = [o.recognition_pts_ms for o in obs_list if o.recognition_pts_ms is not None]

            min_first_pts = min(first_pts_candidates) if first_pts_candidates else None
            max_last_pts = max(last_pts_candidates) if last_pts_candidates else None

            # Select strongest evidence observation
            best_obs = max(obs_list, key=lambda o: (o.consensus_score, o.ocr_confidence))
            rec_pts = best_obs.recognition_pts_ms or (min_first_pts if min_first_pts is not None else 0.0)

            seg_id = f"SEG-{first_obs.vehicle_id}-{cam_id}-{trk_id}"

            segment = CameraObservationSegment(
                segment_id=seg_id,
                vehicle_id=first_obs.vehicle_id,
                registration_number=first_obs.registration_number,
                normalized_registration_number=first_obs.normalized_registration_number,
                camera_id=cam_id,
                track_id=trk_id,
                first_seen_pts_ms=min_first_pts,
                recognition_pts_ms=rec_pts,
                last_seen_pts_ms=max_last_pts,
                source_time=best_obs.source_time,
                source_time_status=best_obs.source_time_status,
                loop_instance=best_obs.loop_instance,
                recognition_status=best_obs.recognition_status,
                consensus_score=best_obs.consensus_score,
                ocr_confidence=best_obs.ocr_confidence,
                camera_name=best_obs.camera_name,
                camera_latitude=best_obs.camera_latitude,
                camera_longitude=best_obs.camera_longitude,
                evidence_image=best_obs.evidence_image,
                evidence_filename_pts_status=best_obs.evidence_filename_pts_status,
                supporting_observation_count=len(obs_list),
                observation_ids=[o.observation_id for o in obs_list],
            )
            segments.append(segment)
            seg_idx += 1

        return segments
