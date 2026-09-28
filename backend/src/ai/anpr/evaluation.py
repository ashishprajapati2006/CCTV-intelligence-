"""ANPR Evaluation, Failure Classification, and Metrics Module.

Implements:
  1. Dataset sample validation (file check, dimensions, ground truth, condition, source).
  2. Failure classification (14 explicit categories: CORRECT, CHARACTER_CONFUSION,
     FALSE_RECOGNITION, OCR_EMPTY, OCR_LOW_CONFIDENCE, INVALID_FORMAT,
     NORMALIZATION_ERROR, PLATE_TOO_SMALL, LOW_PLATE_QUALITY, GLARE, BLUR,
     IMAGE_INVALID, PLATE_MISSING, CONSENSUS_CONFLICT).
  3. Image-level metrics (exact match, normalized match, character accuracy, unreadable, false rec).
  4. Track-level multi-frame consensus evaluation.
  5. Condition-level breakdown with explicit sample counts.
  6. Source-level breakdown (sentinel vs controlled).
  7. Preprocessing variant ablation analysis.
  8. Confidence threshold tradeoff analysis.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.ai.anpr.consensus import MultiFrameConsensusEngine, _levenshtein_distance
from src.ai.anpr.normalizer import normalize_raw_text, normalize_with_audit
from src.ai.anpr.ocr import OCRRecognizer
from src.ai.anpr.schemas import ANPRObservation, OCRResult, TrackANPRConsensus
from src.ai.anpr.validator import validate_indian_plate
from src.common.logging import get_logger

logger = get_logger("anpr_evaluation")

# Standard allowed conditions
ALLOWED_CONDITIONS = {
    "clear",
    "night",
    "blur",
    "glare",
    "small_plate",
    "angled",
    "occluded",
    "low_contrast",
    "motion_blur",
    "compression",
    "unknown",
}

# Explicit failure categories
FAILURE_CATEGORIES = [
    "CORRECT",
    "CHARACTER_CONFUSION",
    "FALSE_RECOGNITION",
    "OCR_EMPTY",
    "OCR_LOW_CONFIDENCE",
    "INVALID_FORMAT",
    "NORMALIZATION_ERROR",
    "PLATE_TOO_SMALL",
    "LOW_PLATE_QUALITY",
    "GLARE",
    "BLUR",
    "IMAGE_INVALID",
    "PLATE_MISSING",
    "CONSENSUS_CONFLICT",
]


@dataclass
class DatasetSampleValidation:
    """Validation report for a single evaluation record before OCR."""
    image_path: str
    exists: bool
    readable: bool
    width: int
    height: int
    file_size_bytes: int
    ground_truth: str
    normalized_ground_truth: str
    condition: str
    source: str
    status: str  # VALID, MISSING_IMAGE, INVALID_IMAGE, MISSING_GROUND_TRUTH, INVALID_LABEL

    @property
    def is_valid(self) -> bool:
        return self.status == "VALID"


@dataclass
class SampleDiagnostic:
    """Detailed per-sample evaluation diagnostic."""
    image_path: str
    camera_id: str
    track_id: str
    condition: str
    source: str
    ground_truth: str
    normalized_ground_truth: str
    raw_ocr: str
    normalized_ocr: str
    ocr_confidence: float
    plate_detection_confidence: float
    plate_quality_score: float
    image_width: int
    image_height: int
    exact_match: bool
    normalized_match: bool
    character_accuracy: float
    failure_reason: str
    processing_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_dataset_sample(
    record: Dict[str, Any],
    dataset_root: Optional[Path] = None,
) -> Tuple[DatasetSampleValidation, Optional[np.ndarray]]:
    """Validate a dataset sample before attempting OCR."""
    img_rel = record.get("image", record.get("image_path", ""))
    gt = record.get("ground_truth", "")
    cond = str(record.get("condition", "unknown")).lower()
    if cond not in ALLOWED_CONDITIONS:
        cond = "unknown"

    source = str(record.get("source", "unknown")).lower()
    if source not in ("sentinel", "controlled", "unknown"):
        source = "unknown"

    if not gt or not isinstance(gt, str) or not gt.strip():
        return DatasetSampleValidation(
            image_path=str(img_rel), exists=False, readable=False, width=0, height=0,
            file_size_bytes=0, ground_truth="", normalized_ground_truth="",
            condition=cond, source=source, status="MISSING_GROUND_TRUTH"
        ), None

    gt_clean = normalize_raw_text(gt)

    # Resolve image path
    p = Path(img_rel)
    if not p.is_absolute() and dataset_root:
        # Check relative to dataset_root, then relative to cwd
        cand_p = dataset_root / img_rel
        if cand_p.exists():
            p = cand_p
        else:
            cand_p2 = Path.cwd() / img_rel
            if cand_p2.exists():
                p = cand_p2
    elif not p.exists():
        cand_p = Path.cwd() / img_rel
        if cand_p.exists():
            p = cand_p

    if not p.exists():
        return DatasetSampleValidation(
            image_path=str(img_rel), exists=False, readable=False, width=0, height=0,
            file_size_bytes=0, ground_truth=gt, normalized_ground_truth=gt_clean,
            condition=cond, source=source, status="MISSING_IMAGE"
        ), None

    file_bytes = p.stat().st_size
    img = cv2.imread(str(p))
    if img is None or img.size == 0:
        return DatasetSampleValidation(
            image_path=str(p), exists=True, readable=False, width=0, height=0,
            file_size_bytes=file_bytes, ground_truth=gt, normalized_ground_truth=gt_clean,
            condition=cond, source=source, status="INVALID_IMAGE"
        ), None

    h, w = img.shape[:2]
    return DatasetSampleValidation(
        image_path=str(p), exists=True, readable=True, width=w, height=h,
        file_size_bytes=file_bytes, ground_truth=gt, normalized_ground_truth=gt_clean,
        condition=cond, source=source, status="VALID"
    ), img


def compute_character_accuracy(ground_truth: str, predicted: str) -> float:
    """Compute character-level accuracy using normalized Levenshtein distance."""
    gt_clean = normalize_raw_text(ground_truth)
    pred_clean = normalize_raw_text(predicted)

    if not gt_clean:
        return 1.0 if not pred_clean else 0.0

    dist = _levenshtein_distance(gt_clean, pred_clean)
    return max(0.0, 1.0 - (dist / float(len(gt_clean))))


def classify_failure(
    ground_truth: str,
    raw_ocr: str,
    normalized_ocr: str,
    ocr_conf: float,
    val: DatasetSampleValidation,
    min_confidence: float = 0.20,
) -> str:
    """Classify the exact failure category for a sample."""
    if not val.exists:
        return "PLATE_MISSING"
    if not val.readable:
        return "IMAGE_INVALID"

    gt_norm = val.normalized_ground_truth

    # If normalized match is achieved, it's correct
    if normalized_ocr == gt_norm and normalized_ocr:
        return "CORRECT"

    # Specific conditions based on physical image size
    if val.width < 35 or val.height < 14:
        return "PLATE_TOO_SMALL"

    if val.condition == "glare":
        return "GLARE"
    if val.condition in ("blur", "motion_blur"):
        return "BLUR"

    # OCR text checks
    if not raw_ocr.strip():
        return "OCR_EMPTY"

    if ocr_conf < min_confidence:
        return "OCR_LOW_CONFIDENCE"

    # Check if raw OCR matched but normalization broke it
    if raw_ocr.strip().upper() == gt_norm and normalized_ocr != gt_norm:
        return "NORMALIZATION_ERROR"

    # Check for minor character confusion (Levenshtein distance <= 2)
    dist = _levenshtein_distance(gt_norm, normalized_ocr)
    if dist <= 2:
        return "CHARACTER_CONFUSION"

    # Format check
    is_valid_fmt, _, _ = validate_indian_plate(normalized_ocr)
    if not is_valid_fmt:
        return "INVALID_FORMAT"

    return "FALSE_RECOGNITION"
