"""Vehicle observation aggregator for Observed Vehicle Database (Step 7).

Aggregates multiple individual observations into a unified ObservedVehicle record.
Enforces the Common-Clock Rule:
  - PTS is camera-local and MUST NOT be compared directly across different cameras.
  - Sighting sequences across cameras are grouped by camera unless source_time is resolved.
"""
from __future__ import annotations

from typing import Dict, List, Optional
from src.observed.models import (
    ObservedVehicle,
    SightingRecord,
    TimelineEntry,
    VehicleObservation,
)
from src.common.logging import get_logger

logger = get_logger("observed_aggregator")


class VehicleAggregator:
    """Aggregates vehicle observations into persistent vehicle identity profiles."""

    @staticmethod
    def _generate_vehicle_id(normalized_reg: str, index: int = 1) -> str:
        """Create deterministic vehicle identifier."""
        return f"VEH-{index:06d}"

    def aggregate_vehicle_observations(
        self,
        normalized_reg: str,
        observations: List[VehicleObservation],
        existing_vehicle: Optional[ObservedVehicle] = None,
        vehicle_index: int = 1,
    ) -> Optional[ObservedVehicle]:
        """Aggregate observations for a single normalized registration number.

        Args:
            normalized_reg: Normalized registration number (e.g., 'GJ01AB1234').
            observations: List of valid VehicleObservation records for this plate.
            existing_vehicle: Existing record if updating in-place.
            vehicle_index: Index number for generating vehicle_id if new.

        Returns:
            ObservedVehicle with aggregated metrics and observation history.
        """
        if not observations:
            return existing_vehicle

        # Distinct cameras
        cameras_seen: List[str] = []
        for o in observations:
            if o.camera_id not in cameras_seen:
                cameras_seen.append(o.camera_id)

        is_single_camera = len(cameras_seen) <= 1

        # Check if source_time is resolved across observations
        all_have_source_time = all(
            o.source_time_status == "RESOLVED" and o.source_time is not None
            for o in observations
        )

        # Ordering strategy:
        # 1. If source_time is resolved, sort by source_time.
        # 2. If single camera, sort by first_seen_pts_ms.
        # 3. If multiple cameras without source_time: group by camera, do NOT compare PTS across cameras!
        if all_have_source_time:
            sorted_obs = sorted(observations, key=lambda o: str(o.source_time))
        elif is_single_camera:
            sorted_obs = sorted(
                observations,
                key=lambda o: (o.first_seen_pts_ms if o.first_seen_pts_ms is not None else 0.0)
            )
        else:
            # Preserve arrival / camera sequence, grouping observations by camera
            # and sorting within each camera by camera-local PTS
            obs_by_cam: Dict[str, List[VehicleObservation]] = {}
            for o in observations:
                obs_by_cam.setdefault(o.camera_id, []).append(o)

            sorted_obs = []
            for cam in cameras_seen:
                cam_obs = sorted(
                    obs_by_cam[cam],
                    key=lambda o: (o.first_seen_pts_ms if o.first_seen_pts_ms is not None else 0.0)
                )
                sorted_obs.extend(cam_obs)

        earliest = sorted_obs[0]
        latest = sorted_obs[-1]

        first_seen = SightingRecord(
            camera_id=earliest.camera_id,
            pts_ms=earliest.first_seen_pts_ms,
            source_time=earliest.source_time,
            source_time_status=earliest.source_time_status,
        )

        last_seen = SightingRecord(
            camera_id=latest.camera_id,
            pts_ms=latest.last_seen_pts_ms,
            source_time=latest.source_time,
            source_time_status=latest.source_time_status,
        )

        # Distinct tracks
        distinct_tracks = set(o.track_id for o in sorted_obs)

        # Consensus scores
        consensus_scores = [o.consensus_score for o in sorted_obs]
        best_score = max(consensus_scores) if consensus_scores else 0.0
        avg_score = (sum(consensus_scores) / float(len(consensus_scores))) if consensus_scores else 0.0

        # Build observation timeline entries
        timeline: List[TimelineEntry] = []
        for o in sorted_obs:
            timeline.append(TimelineEntry(
                camera_id=o.camera_id,
                track_id=o.track_id,
                first_seen_pts_ms=o.first_seen_pts_ms,
                recognition_pts_ms=o.recognition_pts_ms,
                last_seen_pts_ms=o.last_seen_pts_ms,
                source_time=o.source_time,
                source_time_status=o.source_time_status,
                status=o.status,
                consensus_score=o.consensus_score,
                evidence_image=o.evidence_image,
                evidence_filename_pts_status=o.evidence_filename_pts_status,
                ingested_at_utc=o.ingested_at_utc,
            ))

        display_reg = sorted_obs[-1].registration_number or normalized_reg
        veh_id = existing_vehicle.vehicle_id if existing_vehicle else self._generate_vehicle_id(normalized_reg, vehicle_index)

        return ObservedVehicle(
            vehicle_id=veh_id,
            registration_number=display_reg,
            normalized_registration_number=normalized_reg,
            first_seen=first_seen,
            last_seen=last_seen,
            camera_count=len(cameras_seen),
            cameras=cameras_seen,
            observation_count=len(sorted_obs),
            track_count=len(distinct_tracks),
            best_consensus_score=round(best_score, 4),
            average_consensus_score=round(avg_score, 4),
            status="OBSERVED",
            timeline=timeline,
        )
