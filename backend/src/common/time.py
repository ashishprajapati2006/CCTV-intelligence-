"""Timing utilities separating media PTS from local arrival clocks.

CRITICAL ARCHITECTURAL PRINCIPLE:
1. Media PTS (Presentation Time Stamp) is the ONLY authoritative source
   for video timeline, temporal alignment, and multi-camera correlation.
2. Local monotonic clock is used exclusively for arrival diagnostics,
   timeout detection, and transport health monitoring.
3. NEVER compute video timing from CAP_PROP_FPS or frame arrival time.
"""
import time
from datetime import datetime, timezone


def get_monotonic_time() -> float:
    """Return local monotonic time in seconds (immune to system clock changes)."""
    return time.monotonic()


def get_monotonic_ms() -> float:
    """Return local monotonic time in milliseconds."""
    return time.monotonic() * 1000.0


def format_iso8601_now() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def pts_diff_ms(current_pts_ms: float, previous_pts_ms: float) -> float:
    """Calculate the PTS difference between two frames in milliseconds."""
    return current_pts_ms - previous_pts_ms
