"""Plate crop extraction and image preprocessing variants for Step 5.

PREPROCESSING VARIANTS FOR FUTURE OCR (Step 6):
  1. original    - Raw crop from source image (untouched).
  2. grayscale   - BGR to Gray conversion.
  3. upscaled    - Bicubic interpolation 2x scaling (explicitly labeled as interpolation).
  4. contrast    - CLAHE (Contrast Limited Adaptive Histogram Equalization) on L channel.
  5. sharpened   - Unsharp masking filter.
  6. denoised    - Fast Non-Local Means Denoising.

PERSPECTIVE HANDLING:
  Supports optional perspective transformation if 4 corner points are provided.
  If only axis-aligned bbox is present, rectification is skipped without error.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

from src.common.logging import get_logger

logger = get_logger("plate_preprocessor")

_SAFE_CHARS = re.compile(r"[^\w\-]")


def _safe_name(s: str) -> str:
    return _SAFE_CHARS.sub("_", s)


def crop_plate_with_margin(
    frame: np.ndarray,
    plate_bbox: List[int],
    margin_percent: float = 0.10,
) -> Tuple[np.ndarray, List[int]]:
    """Extract plate crop from frame with a small configurable boundary margin.

    Args:
        frame: Full frame image (BGR).
        plate_bbox: [x1, y1, x2, y2] in full frame coords.
        margin_percent: Fractional expansion of width/height (e.g. 0.10 = 10% margin).

    Returns:
        (crop_image, clamped_bbox)
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = plate_bbox

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    mx = int(bw * margin_percent)
    my = int(bh * margin_percent)

    cx1 = max(0, x1 - mx)
    cy1 = max(0, y1 - my)
    cx2 = min(w, x2 + mx)
    cy2 = min(h, y2 + my)

    crop = frame[cy1:cy2, cx1:cx2].copy()
    clamped_bbox = [cx1, cy1, cx2, cy2]
    return crop, clamped_bbox


def generate_preprocessing_variants(
    crop: np.ndarray,
    scale_factor: float = 2.0,
    enable_denoise: bool = True,
) -> Dict[str, np.ndarray]:
    """Generate image preprocessing variants for OCR input preparation.

    Original crop is preserved untouched.

    Returns dict mapping variant_name -> processed_image_array.
    """
    variants: Dict[str, np.ndarray] = {}
    if crop is None or crop.size == 0:
        return variants

    # 1. Original (copy)
    variants["original"] = crop.copy()

    # 2. Grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    variants["grayscale"] = gray.copy()

    # 3. Upscaled (Bicubic interpolation)
    h, w = gray.shape[:2]
    new_w = max(1, int(w * scale_factor))
    new_h = max(1, int(h * scale_factor))
    upscaled = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    variants["upscaled"] = upscaled.copy()

    # 4. Contrast Enhanced (CLAHE on grayscale)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(upscaled)
    variants["contrast"] = contrast.copy()

    # 5. Sharpened (Unsharp Mask on contrast-enhanced)
    gaussian = cv2.GaussianBlur(contrast, (0, 0), 3.0)
    sharpened = cv2.addWeighted(contrast, 1.5, gaussian, -0.5, 0)
    variants["sharpened"] = sharpened.copy()

    # 6. Denoised (optional, using fastNlMeans)
    if enable_denoise:
        try:
            denoised = cv2.fastNlMeansDenoising(contrast, h=7, templateWindowSize=7, searchWindowSize=21)
            variants["denoised"] = denoised
        except Exception as e:
            logger.debug("Denoising failed, skipping variant: %s", e)

    return variants


def warp_perspective(
    crop: np.ndarray,
    corners: np.ndarray,  # 4x2 array of float32 points (top-left, top-right, bottom-right, bottom-left)
    target_width: int = 120,
    target_height: int = 40,
) -> Optional[np.ndarray]:
    """Optional perspective rectification if 4 corner points are available."""
    if corners is None or len(corners) != 4:
        return None
    try:
        dst_pts = np.array(
            [[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_pts)
        warped = cv2.warpPerspective(crop, M, (target_width, target_height))
        return warped
    except Exception as e:
        logger.warning("Perspective warping failed: %s", e)
        return None


def save_plate_crops(
    crop: np.ndarray,
    camera_id: str,
    track_id: str,
    frame_id: int,
    pts_ms: Optional[float],
    plate_index: int,
    snapshot_dir: str,
    generate_variants: bool = True,
) -> Tuple[str, Dict[str, str]]:
    """Save original plate crop and preprocessing variants to structured directories.

    Directories:
      Originals:  data/snapshots/<camera_id>/plates/
      Variants:   data/snapshots/<camera_id>/plates_processed/<variant_name>/

    Filename format:
      <camera_id>_<track_id>_frame<frame_id:06d>_pts<pts:.0f>_plate<plate_index:02d>.jpg

    Returns:
      (original_crop_path, dict of {variant_name: variant_path})
    """
    safe_cam = _safe_name(camera_id)
    safe_track = _safe_name(track_id)
    pts_str = f"{int(pts_ms)}" if pts_ms is not None else "noPTS"

    filename = f"{safe_cam}_{safe_track}_frame{frame_id:06d}_pts{pts_str}_plate{plate_index:02d}.jpg"

    base_snap_dir = Path(snapshot_dir) / safe_cam
    orig_dir = base_snap_dir / "plates"
    proc_base_dir = base_snap_dir / "plates_processed"

    orig_dir.mkdir(parents=True, exist_ok=True)

    orig_path = orig_dir / filename
    cv2.imwrite(str(orig_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    original_crop_path = str(orig_path)

    variant_paths: Dict[str, str] = {}

    if generate_variants:
        variants = generate_preprocessing_variants(crop)
        for var_name, var_img in variants.items():
            if var_name == "original":
                continue
            var_dir = proc_base_dir / var_name
            var_dir.mkdir(parents=True, exist_ok=True)
            var_path = var_dir / filename
            cv2.imwrite(str(var_path), var_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            variant_paths[var_name] = str(var_path)

    return original_crop_path, variant_paths
