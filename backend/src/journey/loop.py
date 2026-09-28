"""Loop and Media-Session Detection for Step 11.1.

Detects PTS reset patterns that indicate a stream loop or new recording session.
Does NOT fabricate loop numbers — loop_instance remains null unless the upstream
pipeline provides it explicitly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.common.logging import get_logger

logger = get_logger("journey_loop")

# A PTS reset is detected when the current PTS is less than this fraction of
# the previous PTS.  Rationale: legitimate forward-play cannot reduce PTS to
# less than 10% of where it was (that would be tens of thousands of ms backwards).
_RESET_RATIO_THRESHOLD = 0.10

# Minimum previous PTS to bother checking (avoids false positives at stream start)
_MIN_PREV_PTS_FOR_RESET_CHECK_MS = 5000.0


@dataclass
class LoopTransitionResult:
    """Result of a PTS-reset / loop-transition check between two consecutive PTS values."""
    loop_transition_detected: bool
    previous_pts_ms: Optional[float]
    current_pts_ms: Optional[float]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LoopDetector:
    """Detects PTS resets indicating stream loops or new recording sessions.

    Algorithm
    ---------
    For each consecutive (prev_pts, curr_pts) pair:
      - If prev_pts >= MIN_PREV_PTS_FOR_RESET_CHECK and
        curr_pts < prev_pts * RESET_RATIO_THRESHOLD → loop transition detected.
      - Otherwise → normal PTS progression.

    The detector DOES NOT assign loop instance numbers.
    loop_instance on observations remains null unless set by the upstream
    ingestion pipeline (Step 4/5).
    """

    def check_transition(
        self,
        previous_pts_ms: Optional[float],
        current_pts_ms: Optional[float],
    ) -> LoopTransitionResult:
        """Check whether two consecutive PTS values indicate a loop reset.

        Parameters
        ----------
        previous_pts_ms:
            PTS of the earlier observation (ms). None if unknown.
        current_pts_ms:
            PTS of the later observation (ms). None if unknown.
        """
        if previous_pts_ms is None or current_pts_ms is None:
            return LoopTransitionResult(
                loop_transition_detected=False,
                previous_pts_ms=previous_pts_ms,
                current_pts_ms=current_pts_ms,
                explanation="Cannot check: one or both PTS values are None.",
            )

        if previous_pts_ms < _MIN_PREV_PTS_FOR_RESET_CHECK_MS:
            return LoopTransitionResult(
                loop_transition_detected=False,
                previous_pts_ms=previous_pts_ms,
                current_pts_ms=current_pts_ms,
                explanation=(
                    f"Previous PTS ({previous_pts_ms} ms) below minimum threshold "
                    f"({_MIN_PREV_PTS_FOR_RESET_CHECK_MS} ms); skipping reset check."
                ),
            )

        threshold = previous_pts_ms * _RESET_RATIO_THRESHOLD
        if current_pts_ms < threshold:
            return LoopTransitionResult(
                loop_transition_detected=True,
                previous_pts_ms=previous_pts_ms,
                current_pts_ms=current_pts_ms,
                explanation=(
                    f"PTS reset detected: previous={previous_pts_ms} ms, "
                    f"current={current_pts_ms} ms "
                    f"(< {_RESET_RATIO_THRESHOLD * 100:.0f}% of previous). "
                    "Likely stream loop or new recording session."
                ),
            )

        return LoopTransitionResult(
            loop_transition_detected=False,
            previous_pts_ms=previous_pts_ms,
            current_pts_ms=current_pts_ms,
            explanation=(
                f"Normal PTS progression: previous={previous_pts_ms} ms, "
                f"current={current_pts_ms} ms."
            ),
        )

    def scan_pts_sequence(
        self,
        pts_sequence: List[Optional[float]],
    ) -> List[LoopTransitionResult]:
        """Scan an ordered list of PTS values for loop transitions.

        Returns a list of LoopTransitionResult for each consecutive pair.
        Length of result = len(pts_sequence) - 1.
        """
        results: List[LoopTransitionResult] = []
        for i in range(len(pts_sequence) - 1):
            result = self.check_transition(pts_sequence[i], pts_sequence[i + 1])
            if result.loop_transition_detected:
                logger.info(
                    "Loop transition detected at index %d: %s",
                    i,
                    result.explanation,
                )
            results.append(result)
        return results

    def scan_observation_segments(
        self,
        segments: List[Any],
    ) -> Tuple[List[LoopTransitionResult], int]:
        """Scan CameraObservationSegment list for PTS resets.

        Only checks same-camera consecutive segments (cross-camera resets are
        expected and do not indicate loops).

        Returns (results, transition_count).
        """
        transitions: List[LoopTransitionResult] = []
        count = 0

        prev_by_cam: Dict[str, Optional[float]] = {}
        for seg in segments:
            cam = getattr(seg, "camera_id", None)
            pts = getattr(seg, "first_seen_pts_ms", None)

            if cam is None:
                continue

            if cam in prev_by_cam:
                result = self.check_transition(prev_by_cam[cam], pts)
                transitions.append(result)
                if result.loop_transition_detected:
                    count += 1

            prev_by_cam[cam] = pts

        return transitions, count
