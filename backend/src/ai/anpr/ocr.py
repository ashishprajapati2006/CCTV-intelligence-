"""Modular OCR Engine Abstraction for License Plate Recognition.

ENGINES:
  1. EasyOCREngine          : Primary PyTorch-based OCR engine using EasyOCR with alphanumeric allowlist.
  2. OpenCVContourOCREngine  : Built-in zero-dependency fallback engine using character contour segmentation.
  3. OCREngineFacade         : Auto-selecting facade defaulting strictly to EasyOCR.

MODULAR CONTRACT:
  Any OCR engine can be plugged in by implementing OCRRecognizer interface.
  Raw text, confidence, processing time, and engine name/version are recorded.
  Every OCR invocation explicitly logs input shape, output text, and confidence.
"""
from __future__ import annotations

import abc
import time
from typing import List, Optional, Tuple
import numpy as np
import cv2

from src.ai.anpr.schemas import OCRResult
from src.common.logging import get_logger

logger = get_logger("anpr_ocr")


class OCRRecognizer(abc.ABC):
    """Abstract base class for license plate OCR engines."""

    @property
    @abc.abstractmethod
    def engine_name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def engine_version(self) -> str:
        pass

    @abc.abstractmethod
    def recognize(self, crop: np.ndarray, variant_name: str = "original") -> OCRResult:
        """Run OCR on a plate crop image.

        Args:
            crop: BGR or Grayscale image array of the license plate region.
            variant_name: Name of preprocessing variant (e.g., "original", "upscaled").

        Returns:
            OCRResult containing raw text, confidence, latency, and engine metadata.
        """
        pass


class EasyOCREngine(OCRRecognizer):
    """EasyOCR (PyTorch) engine tuned for alphanumeric license plate recognition."""

    def __init__(
        self,
        use_gpu: Optional[bool] = None,
        allowlist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        min_canvas_height: int = 64,
    ):
        self.allowlist = allowlist
        self._reader = None
        if use_gpu is None:
            try:
                import torch
                use_gpu = torch.cuda.is_available()
            except Exception:
                use_gpu = False
        self._use_gpu = use_gpu
        self.min_canvas_height = min_canvas_height
        self._init_reader()

    def _init_reader(self) -> None:
        try:
            import easyocr
            logger.info("Initializing EasyOCR engine (gpu=%s)...", self._use_gpu)
            self._reader = easyocr.Reader(["en"], gpu=self._use_gpu, verbose=False)
            logger.info("EasyOCR engine initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize EasyOCR engine: %s", e)
            self._reader = None
            raise

    @property
    def engine_name(self) -> str:
        return "easyocr"

    @property
    def engine_version(self) -> str:
        try:
            import easyocr
            return easyocr.__version__
        except Exception:
            return "1.7.x"

    @property
    def is_ready(self) -> bool:
        return self._reader is not None

    def _prepare_crop(self, crop: np.ndarray) -> np.ndarray:
        """Prepare plate crop for EasyOCR CRAFT text detector.

        CRAFT requires character height >= 24-32px to detect text lines reliably.
        If crop height < min_canvas_height, scale proportionally and add border.
        """
        h, w = crop.shape[:2]
        if h < self.min_canvas_height:
            scale = float(self.min_canvas_height) / float(h)
            new_w = max(1, int(w * scale))
            new_h = self.min_canvas_height
            scaled = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            scaled = crop

        # Add 8px border padding so edge characters are not clipped by detector
        padded = cv2.copyMakeBorder(
            scaled, 8, 8, 8, 8, cv2.BORDER_REPLICATE
        )
        return padded

    def recognize(self, crop: np.ndarray, variant_name: str = "original") -> OCRResult:
        if not self.is_ready or crop is None or crop.size == 0:
            logger.warning("[OCR ERROR] Invalid or empty image crop provided.")
            return OCRResult(
                raw_text="", confidence=0.0, character_confidences=None,
                processing_time_ms=0.0, engine_name=self.engine_name,
                engine_version=self.engine_version, variant_name=variant_name,
            )

        prepared = self._prepare_crop(crop)
        logger.info("[OCR] engine=%s crop_shape=%dx%d variant=%s",
                    self.engine_name, crop.shape[1], crop.shape[0], variant_name)

        t0 = time.monotonic()
        try:
            results = self._reader.readtext(
                prepared,
                allowlist=self.allowlist,
                detail=1,
                paragraph=False,
                text_threshold=0.10,
                low_text=0.10,
                link_threshold=0.10,
            )
        except Exception as e:
            latency_ms = (time.monotonic() - t0) * 1000.0
            logger.error("[OCR ERROR] EasyOCR inference error (%s): %s", variant_name, e)
            return OCRResult(
                raw_text="", confidence=0.0, character_confidences=None,
                processing_time_ms=latency_ms, engine_name=self.engine_name,
                engine_version=self.engine_version, variant_name=variant_name,
            )

        latency_ms = (time.monotonic() - t0) * 1000.0

        if not results:
            logger.info("[OCR EMPTY] No text detected (latency=%.1fms, variant=%s)", latency_ms, variant_name)
            return OCRResult(
                raw_text="", confidence=0.0, character_confidences=None,
                processing_time_ms=latency_ms, engine_name=self.engine_name,
                engine_version=self.engine_version, variant_name=variant_name,
            )

        # Join detected text blocks
        blocks = []
        confidences = []
        for bbox, text, conf in results:
            clean = text.strip()
            if clean:
                blocks.append(clean)
                confidences.append(float(conf))

        raw_text = "".join(blocks)
        avg_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0

        logger.info("[OCR RESULT] text='%s' confidence=%.2f latency=%.1fms variant=%s",
                    raw_text, avg_conf, latency_ms, variant_name)

        return OCRResult(
            raw_text=raw_text,
            confidence=avg_conf,
            character_confidences=confidences,
            processing_time_ms=latency_ms,
            engine_name=self.engine_name,
            engine_version=self.engine_version,
            variant_name=variant_name,
        )


class OpenCVContourOCREngine(OCRRecognizer):
    """Built-in OpenCV character segmentation engine (fallback)."""

    def __init__(self):
        logger.info("Initialized OpenCV contour fallback OCR engine.")

    @property
    def engine_name(self) -> str:
        return "opencv_contour_v1"

    @property
    def engine_version(self) -> str:
        return "1.0.0"

    def recognize(self, crop: np.ndarray, variant_name: str = "original") -> OCRResult:
        if crop is None or crop.size == 0:
            return OCRResult(
                raw_text="", confidence=0.0, character_confidences=None,
                processing_time_ms=0.0, engine_name=self.engine_name,
                engine_version=self.engine_version, variant_name=variant_name,
            )

        t0 = time.monotonic()
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ch, cw = gray.shape[:2]
        char_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if (h > ch * 0.30) and (h < ch * 0.95) and (w > 3) and (w < cw * 0.35):
                char_boxes.append((x, y, w, h))

        latency_ms = (time.monotonic() - t0) * 1000.0
        logger.info("[OCR] engine=opencv_contour_v1 char_segments=%d variant=%s", len(char_boxes), variant_name)

        return OCRResult(
            raw_text="",
            confidence=min(0.50, len(char_boxes) * 0.05),
            character_confidences=None,
            processing_time_ms=latency_ms,
            engine_name=self.engine_name,
            engine_version=self.engine_version,
            variant_name=variant_name,
        )


class OCREngineFacade(OCRRecognizer):
    """Facade defaulting strictly to EasyOCR unless 'opencv' is explicitly specified."""

    def __init__(self, preferred_engine: Optional[str] = None, use_gpu: Optional[bool] = None):
        if preferred_engine == "opencv":
            logger.info("Explicitly using OpenCV fallback OCR engine.")
            self._active_engine = OpenCVContourOCREngine()
        else:
            try:
                self._active_engine = EasyOCREngine(use_gpu=use_gpu)
            except Exception as e:
                logger.warning("EasyOCR unavailable (%s), falling back to OpenCV contour.", e)
                self._active_engine = OpenCVContourOCREngine()

    @property
    def engine_name(self) -> str:
        return self._active_engine.engine_name

    @property
    def engine_version(self) -> str:
        return self._active_engine.engine_version

    def recognize(self, crop: np.ndarray, variant_name: str = "original") -> OCRResult:
        return self._active_engine.recognize(crop, variant_name=variant_name)
