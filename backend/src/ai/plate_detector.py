"""License plate region detector for Step 5.

ARCHITECTURE: Modular, swappable detector interface
  - Strategy 1 (Primary):  YOLO LP model ('license_plate_detector.pt') if present on disk
  - Strategy 2 (Secondary): OpenCV Haar cascade if XML file present
  - Strategy 3 (Guaranteed Fallback): Classical CV Morphological & Sobel Edge Contour Detector
                            (requires zero external model files, ships with opencv)

STEP 5 SCOPE:
  This module locates the plate REGION only.
  It does NOT read, parse, or return plate text.
  OCR belongs to Step 6.

COORDINATE CONTRACT:
  detect() always returns bounding boxes in FULL-FRAME pixel coordinates,
  even when detection is run on a vehicle-ROI crop (offset_x/offset_y applied).

DETECTOR VERSIONS:
  "contour_v1" - Classical Morphological Sobel Contour Detector (always available)
  "haar_v1"    - OpenCV Haar cascade
  "yolo_lp_v1" - YOLO LP model
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.common.logging import get_logger

logger = get_logger("plate_detector")

_HAAR_CASCADE_NAME = "haarcascade_russian_plate_number.xml"
_YOLO_LP_MODEL_PATH = "license_plate_detector.pt"


@dataclass
class PlateDetection:
    """Raw plate detection from the detector, in full-frame coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    detector_version: str

    @property
    def bbox(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height


class _ContourPlateDetector:
    """Classical Computer Vision Morphological + Sobel Edge Plate Detector.

    Uses vertical Sobel edge density and morphological closing to detect rectangular
    high-contrast character plate regions in vehicle crops. Zero dependencies, 100% offline.
    """

    VERSION = "contour_v1"

    def __init__(
        self,
        min_aspect_ratio: float = 1.5,
        max_aspect_ratio: float = 6.5,
        min_area: int = 200,
        max_area_ratio: float = 0.35, # max plate area relative to vehicle crop
    ):
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio

    def detect(
        self,
        crop: np.ndarray,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> List[PlateDetection]:
        if crop is None or crop.size == 0:
            return []

        ch, cw = crop.shape[:2]
        crop_area = ch * cw
        if crop_area == 0:
            return []

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()

        # 1. Morphological Blackhat to reveal dark regions on light background (or vice versa)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)

        # 2. Sobel X to detect vertical edges (character transitions)
        grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_x = np.absolute(grad_x)
        min_val, max_val = np.min(grad_x), np.max(grad_x)
        if max_val > min_val:
            grad_x = (255 * ((grad_x - min_val) / (max_val - min_val))).astype("uint8")
        else:
            grad_x = np.zeros_like(gray, dtype=np.uint8)

        # 3. Blur & Thresholding
        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. Morphological Close to connect vertical edge gaps into solid plate rectangle
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # 5. Find Contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        ideal_aspect = 3.5  # standard Indian plate aspect ratio (e.g. 520mm x 110mm)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            if area < self.min_area or area > (crop_area * self.max_area_ratio):
                continue

            if h == 0:
                continue

            aspect_ratio = w / float(h)
            if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                # Score confidence higher if aspect ratio is close to standard 3.5
                aspect_diff = abs(aspect_ratio - ideal_aspect) / ideal_aspect
                confidence = max(0.40, min(0.85, 0.85 - aspect_diff * 0.40))

                results.append(PlateDetection(
                    x1=x + offset_x,
                    y1=y + offset_y,
                    x2=x + w + offset_x,
                    y2=y + h + offset_y,
                    confidence=confidence,
                    detector_version=self.VERSION,
                ))

        # Sort detections by area / confidence descending
        results.sort(key=lambda d: d.confidence * d.area, reverse=True)
        return results


def _find_haar_cascade() -> Optional[str]:
    """Locate OpenCV Haar cascade XML file if available."""
    env_path = os.environ.get("OPENCV_HAARCASCADES", "")
    if env_path:
        p = Path(env_path) / _HAAR_CASCADE_NAME
        if p.exists(): return str(p)

    try:
        cv2_data = cv2.data.haarcascades
        p = Path(cv2_data) / _HAAR_CASCADE_NAME
        if p.exists(): return str(p)
    except AttributeError:
        pass

    for candidate in [
        Path(cv2.__file__).parent / "data" / _HAAR_CASCADE_NAME,
        Path(cv2.__file__).parent.parent / "cv2" / "data" / _HAAR_CASCADE_NAME,
    ]:
        if candidate.exists():
            return str(candidate)

    return None


class _HaarPlateDetector:
    VERSION = "haar_v1"
    SYNTHETIC_CONFIDENCE = 0.60

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 4, min_size: Tuple[int, int] = (30, 10)):
        cascade_path = _find_haar_cascade()
        if cascade_path is None:
            raise FileNotFoundError(f"OpenCV Haar cascade '{_HAAR_CASCADE_NAME}' not found.")
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from: {cascade_path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, crop: np.ndarray, offset_x: int = 0, offset_y: int = 0) -> List[PlateDetection]:
        if crop is None or crop.size == 0:
            return []
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
        gray = cv2.equalizeHist(gray)
        try:
            detections = self._cascade.detectMultiScale(
                gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors, minSize=self.min_size
            )
        except Exception:
            return []
        results = []
        for (x, y, w, h) in detections:
            results.append(PlateDetection(
                x1=x + offset_x, y1=y + offset_y, x2=x + w + offset_x, y2=y + h + offset_y,
                confidence=self.SYNTHETIC_CONFIDENCE, detector_version=self.VERSION,
            ))
        return results


class _YoloLPDetector:
    VERSION = "yolo_lp_v1"

    def __init__(self, model_path: str, confidence_threshold: float = 0.10):
        from ultralytics import YOLO
        logger.info("Loading YOLO LP model: %s", model_path)
        self._model = YOLO(model_path)
        self._conf = confidence_threshold
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        self._model.predict(dummy, verbose=False, conf=self._conf)
        logger.info("YOLO LP detector ready (conf >= %.2f).", self._conf)

    def detect(self, crop: np.ndarray, offset_x: int = 0, offset_y: int = 0) -> List[PlateDetection]:
        if crop is None or crop.size == 0:
            return []
        try:
            results = self._model.predict(crop, verbose=False, conf=self._conf)
        except Exception:
            return []
        out = []
        if not results or results[0].boxes is None:
            return out
        for box in results[0].boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            out.append(PlateDetection(
                x1=int(xyxy[0]) + offset_x, y1=int(xyxy[1]) + offset_y,
                x2=int(xyxy[2]) + offset_x, y2=int(xyxy[3]) + offset_y,
                confidence=conf, detector_version=self.VERSION,
            ))
        return out


class PlateDetector:
    """Unified plate detector facade.

    Selection strategy:
      1. YOLO LP model if 'license_plate_detector.pt' is present.
      2. OpenCV Haar cascade if XML file present.
      3. Classical CV Morphological & Sobel Contour detector (guaranteed fallback).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.10,
        haar_min_neighbors: int = 3,
        haar_scale_factor: float = 1.1,
    ):
        self._detector = None
        self.confidence_threshold = confidence_threshold

        # 1. Try YOLO LP model first
        lp_path = model_path or _YOLO_LP_MODEL_PATH
        if Path(lp_path).exists():
            try:
                self._detector = _YoloLPDetector(lp_path, confidence_threshold)
                self.detector_version = _YoloLPDetector.VERSION
                logger.info("Using YOLO LP plate detector.")
                return
            except Exception as e:
                logger.warning("YOLO LP model load failed (%s), trying Haar fallback.", e)

        # 2. Try Haar cascade
        try:
            self._detector = _HaarPlateDetector(
                scale_factor=haar_scale_factor, min_neighbors=haar_min_neighbors
            )
            self.detector_version = _HaarPlateDetector.VERSION
            logger.info("Using Haar cascade plate detector.")
            return
        except Exception as e:
            logger.debug("Haar cascade unavailable (%s), using classical contour detector.", e)

        # 3. Guaranteed Fallback: Classical Morphological Sobel Edge Contour Detector
        self._detector = _ContourPlateDetector()
        self.detector_version = _ContourPlateDetector.VERSION
        logger.info("Using classical Morphological Sobel edge plate detector (%s).", self.detector_version)

    @property
    def is_ready(self) -> bool:
        return self._detector is not None

    def detect(
        self,
        crop: np.ndarray,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> Tuple[List[PlateDetection], float]:
        """Run plate detection.

        Returns: (list of PlateDetection in full-frame coords, inference_latency_ms)
        """
        if not self.is_ready or crop is None or crop.size == 0:
            return [], 0.0

        t0 = time.monotonic()
        try:
            detections = self._detector.detect(crop, offset_x, offset_y)
        except Exception as e:
            logger.warning("Plate detection failed: %s", e)
            detections = []
        latency_ms = (time.monotonic() - t0) * 1000.0

        filtered = [d for d in detections if d.confidence >= self.confidence_threshold]
        return filtered, latency_ms
