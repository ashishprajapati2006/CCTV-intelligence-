"""Exponential backoff reconnection strategy for CCTV streams."""
import time
from typing import List, Optional
from src.common.logging import get_logger

logger = get_logger("stream_reconnect")


class ExponentialBackoff:
    """Manages reconnect intervals using a bounded exponential backoff progression.

    Schedule: 2s -> 4s -> 8s -> 16s -> 30s (maximum).
    """

    DEFAULT_STEPS: List[float] = [2.0, 4.0, 8.0, 16.0, 30.0]

    def __init__(
        self,
        steps: Optional[List[float]] = None,
        stable_frames_threshold: int = 30,
    ):
        self.steps = steps or list(self.DEFAULT_STEPS)
        self.current_step_idx: int = 0
        self.attempts: int = 0
        self.consecutive_successful_frames: int = 0
        self.stable_frames_threshold: int = stable_frames_threshold

    @property
    def current_delay(self) -> float:
        """Get the current backoff delay in seconds."""
        idx = min(self.current_step_idx, len(self.steps) - 1)
        return self.steps[idx]

    def record_attempt(self) -> float:
        """Register a reconnect attempt, return the delay to sleep, and advance schedule."""
        self.attempts += 1
        self.consecutive_successful_frames = 0
        delay = self.current_delay
        # Advance step index, capping at maximum
        if self.current_step_idx < len(self.steps) - 1:
            self.current_step_idx += 1
        return delay

    def record_frame_success(self) -> bool:
        """Record a successful frame read.

        If the connection has been stable for `stable_frames_threshold` frames,
        resets the backoff state. Returns True if backoff was reset.
        """
        self.consecutive_successful_frames += 1
        if self.consecutive_successful_frames >= self.stable_frames_threshold:
            if self.current_step_idx > 0 or self.attempts > 0:
                logger.info(
                    "Stream stabilized (%d consecutive frames). Resetting reconnect backoff to %.1fs.",
                    self.consecutive_successful_frames,
                    self.steps[0],
                )
                self.reset()
                return True
        return False

    def reset(self) -> None:
        """Reset backoff to the initial interval."""
        self.current_step_idx = 0
        self.attempts = 0
        self.consecutive_successful_frames = 0

    def wait(self) -> float:
        """Wait for the scheduled backoff delay."""
        delay = self.record_attempt()
        logger.warning(
            "Stream disconnected (attempt #%d). Backing off for %.1f seconds...",
            self.attempts,
            delay,
        )
        time.sleep(delay)
        return delay
