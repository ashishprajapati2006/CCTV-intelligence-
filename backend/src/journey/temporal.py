"""Temporal Models and Validation for Step 11 Vehicle Journey Reconstruction.

Strict Common-Clock Rules:
  1. Camera-local PTS is never compared across distinct cameras.
  2. PTS is never converted to a calendar timestamp by adding to datetime.now().
  3. Global chronological ordering is permitted ONLY when source_time_status == 'RESOLVED'.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple, Union

from src.common.logging import get_logger
from src.journey.models import CameraObservationSegment, CrossCameraObservation

logger = get_logger("journey_temporal")


def parse_source_time(time_str: Optional[str]) -> Optional[datetime]:
    """Parse source_time string into a timezone-aware datetime."""
    if not time_str:
        return None

    formats = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%H:%M:%S",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(time_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue

    logger.warning("Could not parse source_time string: %s", time_str)
    return None


def can_order_globally(items: List[Union[CrossCameraObservation, CameraObservationSegment]]) -> bool:
    """Verify whether all items possess a verified common clock (source_time_status == 'RESOLVED')."""
    if not items:
        return False
    for item in items:
        if item.source_time_status != "RESOLVED" or not item.source_time:
            return False
        if parse_source_time(item.source_time) is None:
            return False
    return True


def calculate_time_delta_seconds(
    from_time_str: Optional[str],
    to_time_str: Optional[str],
) -> Optional[float]:
    """Calculate elapsed seconds between two validated source_time strings."""
    dt1 = parse_source_time(from_time_str)
    dt2 = parse_source_time(to_time_str)
    if dt1 is None or dt2 is None:
        return None
    return (dt2 - dt1).total_seconds()


# ---------------------------------------------------------------------------
# Journey-level and observation-level time status constants (Part 4)
# ---------------------------------------------------------------------------

# Journey-level statuses
JOURNEY_STATUS_OBSERVATION_SEQUENCE_ONLY = "OBSERVATION_SEQUENCE_ONLY"
JOURNEY_STATUS_PARTIAL_JOURNEY = "PARTIAL_JOURNEY"
JOURNEY_STATUS_RECONSTRUCTED = "JOURNEY_RECONSTRUCTED"
JOURNEY_STATUS_TIME_CONFLICT = "TIME_CONFLICT"

# Observation-level source_time_status values
OBS_TIME_NOT_RESOLVED = "NOT_RESOLVED"
OBS_TIME_RESOLVED = "RESOLVED"
OBS_TIME_PARTIAL = "PARTIAL"
OBS_TIME_CONFLICT = "CONFLICT"


# ---------------------------------------------------------------------------
# New helpers (Part 3)
# ---------------------------------------------------------------------------

def validate_source_time(
    source_time: Optional[str],
    source_time_status: Optional[str] = None,
) -> dict:
    """Validate a source_time string from an observation.

    Returns
    -------
    dict with keys:
      valid (bool): True if the string parses as a datetime.
      reason (str): Human-readable explanation.
    """
    if source_time is None:
        return {"valid": False, "reason": "source_time is None"}
    if source_time_status == OBS_TIME_NOT_RESOLVED:
        return {"valid": False, "reason": "source_time_status is NOT_RESOLVED"}
    dt = parse_source_time(source_time)
    if dt is None:
        return {"valid": False, "reason": f"Could not parse source_time string: {source_time!r}"}
    return {"valid": True, "reason": "Parsed successfully"}


def compare_source_times(
    obs_a: Any,
    obs_b: Any,
) -> Optional[int]:
    """Compare source times of two observations.

    Returns
    -------
    -1  if obs_a is earlier than obs_b
     0  if equal
     1  if obs_a is later than obs_b
    None if either observation's source time is unresolved or unparseable.
    """
    status_a = getattr(obs_a, "source_time_status", None) or ""
    status_b = getattr(obs_b, "source_time_status", None) or ""

    if status_a != OBS_TIME_RESOLVED or status_b != OBS_TIME_RESOLVED:
        return None

    time_a = getattr(obs_a, "source_time", None)
    time_b = getattr(obs_b, "source_time", None)

    dt_a = parse_source_time(time_a)
    dt_b = parse_source_time(time_b)

    if dt_a is None or dt_b is None:
        return None

    if dt_a < dt_b:
        return -1
    if dt_a > dt_b:
        return 1
    return 0


def detect_time_conflict(segments: List) -> List[str]:
    """Detect contradictory source times across a list of segments.

    Returns a list of human-readable conflict descriptions (empty if none).
    Only checks segments where source_time_status == RESOLVED.
    """
    conflicts: List[str] = []
    resolved = [
        s for s in segments
        if getattr(s, "source_time_status", None) == OBS_TIME_RESOLVED
        and getattr(s, "source_time", None) is not None
    ]

    if len(resolved) < 2:
        return conflicts  # Need at least 2 resolved times to detect conflicts

    # Parse all
    parsed = []
    for seg in resolved:
        dt = parse_source_time(seg.source_time)
        if dt is not None:
            parsed.append((seg, dt))

    # Check for any timestamps that are identical across distinct cameras
    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            seg_i, dt_i = parsed[i]
            seg_j, dt_j = parsed[j]
            cam_i = getattr(seg_i, "camera_id", "?")
            cam_j = getattr(seg_j, "camera_id", "?")
            if cam_i != cam_j and abs((dt_i - dt_j).total_seconds()) < 0.001:
                conflicts.append(
                    f"Identical source_time on distinct cameras {cam_i} and {cam_j}: {seg_i.source_time}"
                )

    return conflicts
