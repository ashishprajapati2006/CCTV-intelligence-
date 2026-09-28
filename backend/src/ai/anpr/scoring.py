"""Evidence Scoring Engine for Step 6 ANPR.

KEPT SEPARATE (Never collapsed into one single unexplained value):
  - plate_detection_confidence : Detector confidence from Step 5.
  - plate_quality_score        : Image quality heuristic from Step 5.
  - ocr_confidence             : Engine confidence from OCR.
  - format_validity_boost      : Indian plate format validity boost.

COMPOSITE EVIDENCE SCORE:
  Combines the separate metrics into an observation_score used by the consensus engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from src.ai.anpr.schemas import OCRResult, NormalizedOCRResult


@dataclass
class ObservationScoreWeights:
    """Configurable weights for ANPR observation evidence scoring."""
    ocr_confidence: float = 0.40
    format_validity: float = 0.30
    plate_quality: float = 0.15
    plate_detection: float = 0.15


DEFAULT_SCORING_WEIGHTS = ObservationScoreWeights()


def compute_observation_score(
    ocr_result: OCRResult,
    norm_result: NormalizedOCRResult,
    plate_detection_confidence: float,
    plate_quality_score: float,
    weights: ObservationScoreWeights = DEFAULT_SCORING_WEIGHTS,
) -> float:
    """Calculate composite observation evidence score [0.0 - 1.0].

    Preserves independent sub-metrics while computing weighted support score.
    """
    ocr_conf = max(0.0, min(1.0, ocr_result.confidence))
    det_conf = max(0.0, min(1.0, plate_detection_confidence))
    qual_score = max(0.0, min(1.0, plate_quality_score))
    fmt_score = 1.0 if norm_result.is_valid_format else 0.20

    score = (
        weights.ocr_confidence * ocr_conf +
        weights.format_validity * fmt_score +
        weights.plate_quality * qual_score +
        weights.plate_detection * det_conf
    )
    return max(0.0, min(1.0, score))
