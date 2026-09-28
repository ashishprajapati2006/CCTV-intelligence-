"""Spatial Distance, Implied Speed, and Plausibility Analysis for Step 11.

Calculates:
  1. Approximate straight-line distance between cameras (Haversine formula).
  2. Implied straight-line speed when both coordinates and resolved timestamps exist.
  3. Plausibility states: PLAUSIBLE, POSSIBLE, ANOMALOUS, UNKNOWN.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from src.journey.models import (
    CameraObservationSegment,
    JourneyLeg,
    PlausibilityState,
)
from src.journey.temporal import calculate_time_delta_seconds


def haversine_distance_meters(
    lat1: Optional[float], lon1: Optional[float],
    lat2: Optional[float], lon2: Optional[float],
) -> Optional[float]:
    """Calculate approximate straight-line distance between two geographic coordinates in meters.

    Returns None if any coordinate is missing.
    """
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None

    r = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2) + math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(r * c, 1)


class PlausibilityAnalyzer:
    """Evaluates spatial-temporal plausibility between consecutive observation segments."""

    def build_journey_legs(
        self,
        ordered_segments: List[CameraObservationSegment],
    ) -> List[JourneyLeg]:
        """Construct journey legs connecting consecutive ordered observation segments."""
        if len(ordered_segments) < 2:
            return []

        legs: List[JourneyLeg] = []
        for i in range(len(ordered_segments) - 1):
            seg_from = ordered_segments[i]
            seg_to = ordered_segments[i + 1]

            leg_id = f"LEG-{seg_from.vehicle_id}-{i + 1:04d}"

            # Calculate time delta if resolved
            time_delta = None
            time_status = "NOT_RESOLVED"
            if seg_from.source_time_status == "RESOLVED" and seg_to.source_time_status == "RESOLVED":
                time_delta = calculate_time_delta_seconds(seg_from.source_time, seg_to.source_time)
                time_status = "RESOLVED"

            # Calculate straight-line distance (Haversine)
            distance_m = haversine_distance_meters(
                seg_from.camera_latitude, seg_from.camera_longitude,
                seg_to.camera_latitude, seg_to.camera_longitude,
            )
            distance_status = "AVAILABLE" if distance_m is not None else "UNAVAILABLE"

            # Implied speed and plausibility
            speed_kmh, plausibility, reason = self._evaluate_leg_plausibility(
                time_delta, distance_m, time_status, seg_from.camera_id, seg_to.camera_id
            )

            leg = JourneyLeg(
                leg_id=leg_id,
                vehicle_id=seg_from.vehicle_id,
                from_observation_id=seg_from.segment_id,
                to_observation_id=seg_to.segment_id,
                from_camera_id=seg_from.camera_id,
                to_camera_id=seg_to.camera_id,
                from_camera_name=seg_from.camera_name,
                to_camera_name=seg_to.camera_name,
                # Source time passthrough (Part 5, 9)
                from_source_time=seg_from.source_time,
                to_source_time=seg_to.source_time,
                time_delta_seconds=time_delta,
                # Straight-line distance (Part 8) — never labelled "road distance"
                straight_line_distance_m=distance_m,
                distance_status=distance_status,
                distance_label="STRAIGHT_LINE_DISTANCE",
                implied_speed_kmh=speed_kmh,
                implied_speed_label="IMPLIED STRAIGHT-LINE SPEED",
                source_time_status=time_status,
                plausibility=plausibility,
                plausibility_reason=reason,
            )
            legs.append(leg)

        return legs

    @staticmethod
    def _evaluate_leg_plausibility(
        time_delta_sec: Optional[float],
        distance_m: Optional[float],
        time_status: str,
        from_cam: str,
        to_cam: str,
    ) -> Tuple[Optional[float], PlausibilityState, str]:
        """Classify plausibility based on time delta and distance."""
        # Handle same-camera repeat sightings
        if from_cam == to_cam:
            if time_delta_sec is not None and time_delta_sec < 0:
                return None, PlausibilityState.ANOMALOUS, "Negative time delta on same camera"
            return 0.0, PlausibilityState.PLAUSIBLE, "Consecutive observation on same camera"

        # If time or coordinates are unresolved -> UNKNOWN
        if time_status != "RESOLVED" or time_delta_sec is None:
            if distance_m is not None:
                return None, PlausibilityState.UNKNOWN, "Source time unresolved; distance known but speed cannot be calculated"
            return None, PlausibilityState.UNKNOWN, "Source time unresolved and camera coordinates unavailable"

        if distance_m is None:
            return None, PlausibilityState.UNKNOWN, "Source time resolved, but camera coordinates unavailable"

        # Both time and distance exist
        if time_delta_sec <= 0:
            return None, PlausibilityState.ANOMALOUS, f"Non-positive time delta ({time_delta_sec}s) between distinct cameras"

        speed_kmh = round((distance_m / 1000.0) / (time_delta_sec / 3600.0), 1)

        if speed_kmh > 180.0:
            return speed_kmh, PlausibilityState.ANOMALOUS, f"Implied straight-line speed ({speed_kmh} km/h) exceeds 180 km/h threshold"
        elif speed_kmh > 120.0:
            return speed_kmh, PlausibilityState.POSSIBLE, f"High implied straight-line speed ({speed_kmh} km/h)"
        else:
            return speed_kmh, PlausibilityState.PLAUSIBLE, f"Plausible implied straight-line speed ({speed_kmh} km/h)"
