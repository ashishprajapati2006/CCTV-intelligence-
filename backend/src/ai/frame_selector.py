"""Frame quality scoring for vehicle candidate selection (Steps 4 and 5).

QUALITY SCORING PHILOSOPHY:
  Scores are heuristic ranking metrics, NOT physical image quality measurements.
  They exist to rank candidate frames for plate detection input.
  Do not interpret them as ground truth.

Metrics:
  sharpness  : Laplacian variance on vehicle crop (higher = less blur)
  brightness : Mean pixel luminance (proximity to ideal)
  contrast   : RMS contrast of crop (higher = more detail)
  area       : Vehicle bbox pixel area (larger = more plate detail available)
  truncation : Fraction of bbox within frame (1.0 = fully inside)
  quality    : Weighted composite of the above, configurable

STEP 5 NOTE:
  plate_region_visibility is left as a future hook.
  It can be added in Step 6 once plate detection is reliable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

import cv2
import numpy as np

from src.ai.schemas import FrameCandidate, VehicleTrack, VehicleFrameCandidate
from src.common.logging import get_logger

logger = get_logger("frame_selector")

_SAFE_CHARS = re.compile(r"[^\w\-]")


def _safe_name(s: str) -> str:
    return _SAFE_CHARS.sub("_", s)


# ─── Individual metric functions ─────────────────────────────────────────────

def compute_sharpness(crop: np.ndarray) -> float:
    """Laplacian variance -- low value indicates motion blur or out-of-focus."""
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(crop: np.ndarray) -> float:
    """Mean pixel luminance (0-255). Extreme values indicate exposure problems."""
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(gray.mean())


def compute_contrast(crop: np.ndarray) -> float:
    """RMS contrast of the crop. Low value = flat, low-detail image."""
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    mean = gray.mean()
    return float(np.sqrt(((gray.astype(np.float32) - mean) ** 2).mean()))


def compute_truncation(bbox: List[int], frame_h: int, frame_w: int) -> float:
    """Fraction of the bbox that falls inside the frame boundary.

    Returns 1.0 when the bbox is fully inside the frame.
    Returns 0.0 when the bbox is completely outside.
    """
    if not bbox or len(bbox) != 4:
        return 0.0
    x1, y1, x2, y2 = bbox
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    if box_area == 0:
        return 0.0
    cx1 = max(0, x1)
    cy1 = max(0, y1)
    cx2 = min(frame_w, x2)
    cy2 = min(frame_h, y2)
    inside_area = max(0, cx2 - cx1) * max(0, cy2 - cy1)
    return inside_area / box_area


# ─── Normalisation helpers ────────────────────────────────────────────────────

def _norm_sharpness(v: float, cap: float = 1000.0) -> float:
    return min(v / cap, 1.0)


def _norm_area(v: int, cap: int = 150_000) -> float:
    return min(v / cap, 1.0)


def _norm_brightness(v: float, ideal: float = 128.0) -> float:
    """Score = 1.0 when brightness == ideal; falls off linearly."""
    return max(0.0, 1.0 - abs(v - ideal) / ideal)


def _norm_contrast(v: float, cap: float = 64.0) -> float:
    return min(v / cap, 1.0)


# ─── Composite scoring ────────────────────────────────────────────────────────

@dataclass
class FrameQualityWeights:
    """Configurable weights for vehicle frame quality scoring.

    All weights are applied AFTER individual metrics are normalised to [0, 1].
    Weights need not sum to 1.0; the composite is NOT re-normalised.
    """
    sharpness:   float = 0.35
    area:        float = 0.25
    brightness:  float = 0.15
    contrast:    float = 0.15
    truncation:  float = 0.10


DEFAULT_WEIGHTS = FrameQualityWeights()


def score_frame(
    sharpness: float,
    bbox_area: int,
    brightness: float,
    w_sharpness: float = 0.5,
    w_area: float = 0.3,
    w_brightness: float = 0.2,
    brightness_ideal: float = 128.0,
) -> float:
    """Step 4 composite quality score (kept for backward compatibility).

    Weights: sharpness 50%, bbox area 30%, brightness 20%.
    """
    norm_sharpness = _norm_sharpness(sharpness)
    norm_area = _norm_area(bbox_area)
    norm_brightness = _norm_brightness(brightness, ideal=brightness_ideal)
    return w_sharpness * norm_sharpness + w_area * norm_area + w_brightness * norm_brightness


def score_frame_extended(
    sharpness: float,
    bbox_area: int,
    brightness: float,
    contrast: float,
    truncation: float,
    weights: FrameQualityWeights = DEFAULT_WEIGHTS,
) -> tuple:
    """Extended vehicle frame quality score (Step 5).

    Returns:
        (composite_score, norm_sharpness, norm_brightness, norm_contrast,
         norm_area, truncation)
    """
    ns = _norm_sharpness(sharpness)
    na = _norm_area(bbox_area)
    nb = _norm_brightness(brightness)
    nc = _norm_contrast(contrast)
    nt = float(truncation)  # already in [0,1]

    composite = (
        weights.sharpness  * ns +
        weights.area       * na +
        weights.brightness * nb +
        weights.contrast   * nc +
        weights.truncation * nt
    )
    return composite, ns, nb, nc, na, nt


# ─── Frame evaluation ─────────────────────────────────────────────────────────

def evaluate_frame(
    frame: np.ndarray,
    bbox: List[int],
    media_pts_ms: Optional[float],
    local_receive_monotonic: float,
    frame_sequence: int,
) -> FrameCandidate:
    """Score a frame against a vehicle bounding box (Step 4 API, unchanged)."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    crop = frame[y1c:y2c, x1c:x2c]

    sharpness = compute_sharpness(crop)
    brightness = compute_brightness(crop)
    bbox_area = max(0, x2c - x1c) * max(0, y2c - y1c)
    quality = score_frame(sharpness, bbox_area, brightness)

    return FrameCandidate(
        frame_sequence=frame_sequence,
        media_pts_ms=media_pts_ms,
        local_receive_monotonic=local_receive_monotonic,
        sharpness=sharpness,
        bbox_area=bbox_area,
        brightness=brightness,
        quality_score=quality,
    )


def evaluate_frame_extended(
    camera_id: str,
    track_id: str,
    vehicle_class: str,
    detection_confidence: float,
    frame: np.ndarray,
    bbox: List[int],
    media_pts_ms: Optional[float],
    local_receive_monotonic: float,
    frame_sequence: int,
    weights: FrameQualityWeights = DEFAULT_WEIGHTS,
    image_path: Optional[str] = None,
) -> VehicleFrameCandidate:
    """Step 5 extended vehicle frame quality evaluation.

    Returns a VehicleFrameCandidate with rich per-metric scores.
    Never raises -- returns zeroed candidate on any crop failure.
    """
    has_valid_pts = (media_pts_ms is not None and media_pts_ms > 0)

    try:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        crop = frame[y1c:y2c, x1c:x2c]

        sharpness = compute_sharpness(crop)
        brightness = compute_brightness(crop)
        contrast = compute_contrast(crop)
        truncation = compute_truncation(bbox, h, w)
        bbox_area = max(0, x2c - x1c) * max(0, y2c - y1c)

        composite, ns, nb, nc, na, nt = score_frame_extended(
            sharpness, bbox_area, brightness, contrast, truncation, weights
        )
    except Exception as e:
        logger.warning(
            "[%s/%s] Frame quality evaluation failed on frame %d: %s",
            camera_id, track_id, frame_sequence, e,
        )
        composite, ns, nb, nc, na, nt = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    return VehicleFrameCandidate(
        camera_id=camera_id,
        track_id=track_id,
        frame_id=frame_sequence,
        pts_ms=media_pts_ms,
        has_valid_pts=has_valid_pts,
        local_receive_monotonic=local_receive_monotonic,
        bbox=list(bbox),
        vehicle_class=vehicle_class,
        detection_confidence=detection_confidence,
        sharpness_score=ns,
        brightness_score=nb,
        contrast_score=nc,
        area_score=na,
        truncation_score=nt,
        vehicle_quality_score=composite,
        image_path=image_path,
    )


# ─── Candidate pool management ────────────────────────────────────────────────

def update_best_frames(
    track: VehicleTrack,
    candidate: FrameCandidate,
    max_best_frames: int,
) -> None:
    """Keep only the top-N best candidates by quality_score (Step 4 API)."""
    track.best_frames.append(candidate)
    track.best_frames.sort(key=lambda fc: fc.quality_score, reverse=True)
    if len(track.best_frames) > max_best_frames:
        track.best_frames = track.best_frames[:max_best_frames]


def select_top_vehicle_frames(
    candidates: List[VehicleFrameCandidate],
    top_n: int,
) -> List[VehicleFrameCandidate]:
    """Return top-N VehicleFrameCandidates ranked by vehicle_quality_score."""
    return sorted(candidates, key=lambda c: c.vehicle_quality_score, reverse=True)[:top_n]


# ─── Snapshot saving ──────────────────────────────────────────────────────────

def save_best_frames(
    track: VehicleTrack,
    frame_data: Dict[int, np.ndarray],
    snapshot_dir: str,
    camera_id: str,
) -> None:
    """Write best-candidate images to disk (Step 4 API, unchanged)."""
    out_dir = Path(snapshot_dir) / _safe_name(camera_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    for candidate in track.best_frames:
        if candidate.file_path:
            continue
        img = frame_data.get(candidate.frame_sequence)
        if img is None:
            continue
        pts_str = f"{int(candidate.media_pts_ms)}" if candidate.media_pts_ms is not None else "noPTS"
        fname = (
            f"{_safe_name(camera_id)}_{_safe_name(track.track_id)}"
            f"_frame{candidate.frame_sequence:06d}_pts{pts_str}.jpg"
        )
        fpath = str(out_dir / fname)
        try:
            cv2.imwrite(fpath, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            candidate.file_path = fpath
        except Exception as e:
            logger.warning("Could not save snapshot %s: %s", fpath, e)
