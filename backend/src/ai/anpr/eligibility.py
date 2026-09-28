"""Candidate eligibility evaluator and rejection diagnostics for Step 6 ANPR.

REJECTION REASONS:
  - NONE                     : Eligible for OCR
  - MISSING_CROP             : Image file does not exist on disk
  - INVALID_IMAGE            : Image file cannot be decoded by OpenCV
  - INVALID_BBOX             : Bounding box coordinates non-positive or inverted
  - TOO_SMALL                : Width/height/area below minimum readable thresholds
  - LOW_DETECTION_CONFIDENCE : Step 5 plate detector confidence below threshold
  - LOW_IMAGE_QUALITY        : Step 5 plate quality score below threshold
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2

from src.common.logging import get_logger

logger = get_logger("anpr_eligibility")


@dataclass
class EligibilityResult:
    """Eligibility decision for a single plate candidate crop."""
    candidate_id: str
    track_id: str
    frame_id: int
    image_path: str
    plate_width: int
    plate_height: int
    plate_area: int
    plate_aspect_ratio: float
    plate_detection_confidence: float
    plate_quality_score: float
    is_eligible: bool
    rejection_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "track_id": self.track_id,
            "frame_id": self.frame_id,
            "image_path": self.image_path,
            "plate_width": self.plate_width,
            "plate_height": self.plate_height,
            "plate_area": self.plate_area,
            "plate_aspect_ratio": round(self.plate_aspect_ratio, 2),
            "plate_detection_confidence": round(self.plate_detection_confidence, 4),
            "plate_quality_score": round(self.plate_quality_score, 4),
            "is_eligible": self.is_eligible,
            "rejection_reason": self.rejection_reason,
        }


class CandidateEligibilityEvaluator:
    """Evaluates candidate plate crops against physical and quality criteria before OCR."""

    def __init__(
        self,
        min_plate_width: int = 8,
        min_plate_height: int = 4,
        min_plate_area: int = 32,
        min_detection_confidence: float = 0.05,
        min_quality_score: float = 0.05,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 10.0,
    ):
        self.min_plate_width = min_plate_width
        self.min_plate_height = min_plate_height
        self.min_plate_area = min_plate_area
        self.min_detection_confidence = min_detection_confidence
        self.min_quality_score = min_quality_score
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

    def evaluate_candidate(
        self,
        candidate_dict: Dict[str, Any],
        image_path: Optional[str] = None,
    ) -> EligibilityResult:
        cand_id = str(candidate_dict.get("candidate_id", candidate_dict.get("frame_id", "?")))
        track_id = str(candidate_dict.get("track_id", "?"))
        frame_id = int(candidate_dict.get("frame_id", 0))

        img_path = image_path or candidate_dict.get("original_crop_path", "")
        det_conf = float(candidate_dict.get("plate_confidence", 0.0))
        qual_score = float(candidate_dict.get("plate_quality_score", 0.0))

        # Check bbox structure
        bbox = candidate_dict.get("plate_bbox", [])
        if not bbox or len(bbox) != 4:
            return EligibilityResult(
                candidate_id=cand_id, track_id=track_id, frame_id=frame_id,
                image_path=str(img_path), plate_width=0, plate_height=0, plate_area=0,
                plate_aspect_ratio=0.0, plate_detection_confidence=det_conf,
                plate_quality_score=qual_score, is_eligible=False,
                rejection_reason="INVALID_BBOX"
            )

        x1, y1, x2, y2 = bbox
        pw = max(0, x2 - x1)
        ph = max(0, y2 - y1)
        area = pw * ph
        aspect_ratio = (pw / float(ph)) if ph > 0 else 0.0

        # Dimension checks
        if pw <= 0 or ph <= 0:
            return EligibilityResult(
                candidate_id=cand_id, track_id=track_id, frame_id=frame_id,
                image_path=str(img_path), plate_width=pw, plate_height=ph, plate_area=area,
                plate_aspect_ratio=aspect_ratio, plate_detection_confidence=det_conf,
                plate_quality_score=qual_score, is_eligible=False,
                rejection_reason="INVALID_BBOX"
            )

        if pw < self.min_plate_width or ph < self.min_plate_height or area < self.min_plate_area:
            return EligibilityResult(
                candidate_id=cand_id, track_id=track_id, frame_id=frame_id,
                image_path=str(img_path), plate_width=pw, plate_height=ph, plate_area=area,
                plate_aspect_ratio=aspect_ratio, plate_detection_confidence=det_conf,
                plate_quality_score=qual_score, is_eligible=False,
                rejection_reason="TOO_SMALL"
            )

        # Confidence checks
        if det_conf < self.min_detection_confidence:
            return EligibilityResult(
                candidate_id=cand_id, track_id=track_id, frame_id=frame_id,
                image_path=str(img_path), plate_width=pw, plate_height=ph, plate_area=area,
                plate_aspect_ratio=aspect_ratio, plate_detection_confidence=det_conf,
                plate_quality_score=qual_score, is_eligible=False,
                rejection_reason="LOW_DETECTION_CONFIDENCE"
            )

        if qual_score < self.min_quality_score:
            return EligibilityResult(
                candidate_id=cand_id, track_id=track_id, frame_id=frame_id,
                image_path=str(img_path), plate_width=pw, plate_height=ph, plate_area=area,
                plate_aspect_ratio=aspect_ratio, plate_detection_confidence=det_conf,
                plate_quality_score=qual_score, is_eligible=False,
                rejection_reason="LOW_IMAGE_QUALITY"
            )

        # File presence & readability check
        if not img_path:
            return EligibilityResult(
                candidate_id=cand_id, track_id=track_id, frame_id=frame_id,
                image_path="", plate_width=pw, plate_height=ph, plate_area=area,
                plate_aspect_ratio=aspect_ratio, plate_detection_confidence=det_conf,
                plate_quality_score=qual_score, is_eligible=False,
                rejection_reason="MISSING_CROP"
            )

        resolved_p = Path(img_path)
        if not resolved_p.exists():
            resolved_p = Path.cwd() / img_path

        if not resolved_p.exists():
            return EligibilityResult(
                candidate_id=cand_id, track_id=track_id, frame_id=frame_id,
                image_path=str(img_path), plate_width=pw, plate_height=ph, plate_area=area,
                plate_aspect_ratio=aspect_ratio, plate_detection_confidence=det_conf,
                plate_quality_score=qual_score, is_eligible=False,
                rejection_reason="MISSING_CROP"
            )

        # Eligible
        return EligibilityResult(
            candidate_id=cand_id, track_id=track_id, frame_id=frame_id,
            image_path=str(resolved_p), plate_width=pw, plate_height=ph, plate_area=area,
            plate_aspect_ratio=aspect_ratio, plate_detection_confidence=det_conf,
            plate_quality_score=qual_score, is_eligible=True,
            rejection_reason="NONE"
        )
