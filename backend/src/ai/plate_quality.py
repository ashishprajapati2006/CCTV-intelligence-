"""Plate quality scoring and plate bounding-box validation for Step 5.

SCORES AND VALIDATION:
  - Bounding box sanity validation (bounds, min area, min aspect ratio, confidence).
  - Individual quality metrics for license plate crops (sharpness, brightness, contrast).
  - Composite plate quality score (0.0 to 1.0) for ranking OCR candidates in Step 6.

NOTE:
  Plate quality score represents image quality / clarity heuristic for OCR suitability,
  NOT OCR confidence or text readability guarantees.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import cv2

from src.common.logging import get_logger

logger = get_logger("plate_quality")


@dataclass
class PlateQualityMetrics:
    """Detailed plate quality breakdown."""
    sharpness: float         # raw Laplacian variance
    brightness: float        # mean luminance (0-255)
    contrast: float          # RMS contrast
    sharpness_score: float   # normalized 0-1
    brightness_score: float  # normalized 0-1
    contrast_score: float    # normalized 0-1
    composite_score: float   # weighted quality score 0-1


def validate_plate_bbox(
    plate_bbox: List[int],
    frame_h: int,
    frame_w: int,
    plate_confidence: float,
    min_confidence: float = 0.35,
    min_area: int = 150,          # minimum pixel area for a plate crop
    min_width: int = 15,
    min_height: int = 8,
    max_aspect_ratio: float = 10.0, # width/height max ratio
) -> Tuple[bool, Optional[str]]:
    """Validate a proposed plate bounding box against boundaries and physical sanity.

    Args:
        plate_bbox: [x1, y1, x2, y2] in full-frame coordinates.
        frame_h: Height of full frame.
        frame_w: Width of full frame.
        plate_confidence: Confidence score reported by detector.
        min_confidence: Configurable confidence threshold.
        min_area: Minimum acceptable pixel area.
        min_width: Minimum pixel width.
        min_height: Minimum pixel height.
        max_aspect_ratio: Max aspect ratio (width/height).

    Returns:
        (is_valid: bool, rejection_reason: Optional[str])
    """
    if not plate_bbox or len(plate_bbox) != 4:
        return False, "invalid_bbox_structure"

    x1, y1, x2, y2 = plate_bbox
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return False, "non_positive_dimensions"

    if x1 < 0 or y1 < 0 or x2 > frame_w or y2 > frame_h:
        return False, "out_of_frame_bounds"

    if plate_confidence < min_confidence:
        return False, f"low_confidence_{plate_confidence:.2f}"

    area = w * h
    if area < min_area:
        return False, f"too_small_area_{area}px"

    if w < min_width or h < min_height:
        return False, f"too_small_dim_{w}x{h}"

    aspect_ratio = w / float(h)
    if aspect_ratio > max_aspect_ratio or aspect_ratio < 0.5:
        return False, f"unrealistic_aspect_ratio_{aspect_ratio:.2f}"

    return True, None


def compute_plate_sharpness(crop: np.ndarray) -> float:
    """Laplacian variance on plate crop."""
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_plate_brightness(crop: np.ndarray) -> float:
    """Mean pixel luminance of plate crop."""
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(gray.mean())


def compute_plate_contrast(crop: np.ndarray) -> float:
    """RMS contrast of plate crop."""
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    mean = gray.mean()
    return float(np.sqrt(((gray.astype(np.float32) - mean) ** 2).mean()))


def score_plate(
    crop: np.ndarray,
    w_sharpness: float = 0.50,
    w_brightness: float = 0.25,
    w_contrast: float = 0.25,
    sharpness_cap: float = 500.0,
    ideal_brightness: float = 128.0,
    contrast_cap: float = 64.0,
) -> PlateQualityMetrics:
    """Calculate plate-level quality score and metrics breakdown.

    Returns PlateQualityMetrics dataclass.
    """
    if crop is None or crop.size == 0:
        return PlateQualityMetrics(
            sharpness=0.0,
            brightness=0.0,
            contrast=0.0,
            sharpness_score=0.0,
            brightness_score=0.0,
            contrast_score=0.0,
            composite_score=0.0,
        )

    sharpness = compute_plate_sharpness(crop)
    brightness = compute_plate_brightness(crop)
    contrast = compute_plate_contrast(crop)

    # Normalization
    ns = min(sharpness / sharpness_cap, 1.0)
    nb = max(0.0, 1.0 - abs(brightness - ideal_brightness) / ideal_brightness)
    nc = min(contrast / contrast_cap, 1.0)

    composite = (w_sharpness * ns) + (w_brightness * nb) + (w_contrast * nc)

    return PlateQualityMetrics(
        sharpness=sharpness,
        brightness=brightness,
        contrast=contrast,
        sharpness_score=ns,
        brightness_score=nb,
        contrast_score=nc,
        composite_score=composite,
    )
