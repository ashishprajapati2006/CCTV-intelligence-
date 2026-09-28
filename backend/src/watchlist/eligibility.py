"""Match Eligibility Evaluator for Step 9 Real-Time Watchlist Matching.

Filters raw ANPR/CCTV observations before watchlist lookup to prevent low-confidence,
uncertain, or invalid plates from triggering automated watchlist matching.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.watchlist.normalizer import (
    normalize_watchlist_registration,
    validate_watchlist_registration,
)


@dataclass
class MatchEligibilityConfig:
    """Configurable operating thresholds for automated watchlist matching eligibility."""
    min_consensus_score: float = 0.80
    min_ocr_confidence: float = 0.70
    min_plate_detection_confidence: float = 0.50
    min_plate_quality_score: float = 0.40
    required_recognition_statuses: List[str] = field(default_factory=lambda: ["CONFIRMED"])
    allow_probable: bool = False  # When True, accepts "PROBABLE" recognitions
    allow_fuzzy_candidates: bool = False  # When True, permits near-format OCR substitutions for human review

    def __post_init__(self):
        if self.allow_probable and "PROBABLE" not in self.required_recognition_statuses:
            self.required_recognition_statuses.append("PROBABLE")


class MatchEligibilityEvaluator:
    """Evaluates whether an ANPR vehicle observation qualifies for watchlist lookup."""

    def __init__(self, config: Optional[MatchEligibilityConfig] = None):
        self.config = config or MatchEligibilityConfig()

    def evaluate(self, observation: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluate observation eligibility.

        Args:
            observation: Dict containing observation attributes.

        Returns:
            (is_eligible: bool, reason: str)
            Where reason is an empty string if eligible, or a structured rejection reason.
        """
        # 1. Check Registration Presence
        reg = observation.get("normalized_registration_number") or observation.get("registration_number")
        if not reg:
            return False, "MISSING_REGISTRATION"

        norm_reg = normalize_watchlist_registration(reg)
        if not norm_reg:
            return False, "MISSING_REGISTRATION"

        # 2. Check Recognition Status (CONFIRMED vs UNCERTAIN/UNREADABLE)
        rec_status = observation.get("recognition_status") or observation.get("status")
        if rec_status not in self.config.required_recognition_statuses:
            return False, "UNCERTAIN_RECOGNITION"

        # 3. Check Registration Format Validity
        is_valid_format = validate_watchlist_registration(norm_reg)
        if not is_valid_format:
            # If fuzzy review candidate mode is explicitly enabled, allow alphanumeric strings of length 8-11
            if self.config.allow_fuzzy_candidates and len(norm_reg) in (8, 9, 10, 11) and norm_reg.isalnum():
                pass
            else:
                return False, "INVALID_REGISTRATION"

        # 4. Check Consensus Score
        consensus = float(observation.get("consensus_score") or 0.0)
        if consensus < self.config.min_consensus_score:
            return False, "LOW_CONSENSUS"

        # 5. Check OCR Confidence
        ocr_conf = float(observation.get("ocr_confidence") or 0.0)
        if ocr_conf < self.config.min_ocr_confidence:
            return False, "LOW_OCR_CONFIDENCE"

        # 6. Check Plate Quality Score (if present)
        quality = observation.get("plate_quality_score")
        if quality is not None and float(quality) < self.config.min_plate_quality_score:
            return False, "LOW_IMAGE_QUALITY"

        return True, "ELIGIBLE"
