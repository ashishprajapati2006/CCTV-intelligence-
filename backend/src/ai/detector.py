"""Vehicle detector abstraction using YOLOv8n (ultralytics).

MODEL CHOICE — YOLOv8n (nano):
  - Pre-trained on COCO-128 → supports car, truck, bus, motorcycle, bicycle
  - Smallest & fastest YOLO variant for CPU-only deployments
  - Single .pt file (~6 MB), auto-downloaded on first run to ultralytics cache
  - Replaceable: swap to yolov8s/m/l/x or any YOLO11 model via config

TIMING RULE:
  Detector measures its own wall-clock inference latency (performance).
  It does NOT touch or modify media PTS from the video stream.
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple
import numpy as np

from src.ai.schemas import Detection
from src.common.logging import get_logger

logger = get_logger("vehicle_detector")

# COCO class IDs that represent vehicles (indices from COCO 80-class list)
VEHICLE_CLASS_IDS = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Additional COCO classes to include if the model detects them
EXTRA_VEHICLE_CLASS_IDS: dict = {}   # e.g. {8: "boat"} if desired


class VehicleDetector:
    """Wraps a YOLO model and returns vehicle-filtered detections.

    Usage::

        detector = VehicleDetector()          # loads yolov8n automatically
        detections = detector.detect(frame)   # List[Detection]
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.40,
        device: Optional[str] = None,
        extra_class_ids: Optional[dict] = None,
    ):
        """
        Args:
            model_name: Ultralytics model identifier or path to .pt file.
                        Default "yolov8n.pt" is auto-downloaded (~6 MB).
            confidence_threshold: Minimum detection confidence 0–1.
            device: "cpu", "cuda:0", or None for automatic selection.
            extra_class_ids: Additional {class_id: label} pairs beyond defaults.
        """
        self.confidence_threshold = confidence_threshold
        self.device = device or "cpu"

        self._all_vehicle_ids = dict(VEHICLE_CLASS_IDS)
        if extra_class_ids:
            self._all_vehicle_ids.update(extra_class_ids)
        if EXTRA_VEHICLE_CLASS_IDS:
            self._all_vehicle_ids.update(EXTRA_VEHICLE_CLASS_IDS)

        self._model = None
        self._model_name = model_name
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO model (auto-downloads weights on first run)."""
        try:
            from ultralytics import YOLO
            logger.info("Loading detector model: %s on device: %s", self._model_name, self.device)
            self._model = YOLO(self._model_name)
            # Warm up
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model.predict(dummy, verbose=False, device=self.device)
            logger.info("Detector model ready (confidence threshold: %.2f).", self.confidence_threshold)
        except Exception as e:
            logger.error("Failed to load detector model '%s': %s", self._model_name, e)
            raise

    def detect(
        self,
        frame: np.ndarray,
        media_pts_ms: Optional[float] = None,
        local_receive_monotonic: Optional[float] = None,
        frame_sequence: int = 0,
    ) -> Tuple[List[Detection], float]:
        """Run vehicle detection on a single BGR frame.

        Args:
            frame: OpenCV BGR image (numpy ndarray).
            media_pts_ms: Container PTS for this frame (passed through to results).
            local_receive_monotonic: Arrival clock for diagnostics.
            frame_sequence: Sequential frame counter.

        Returns:
            Tuple of (list of Detection, inference_latency_ms)
        """
        if self._model is None:
            return [], 0.0

        t0 = time.monotonic()
        try:
            results = self._model.predict(
                frame,
                verbose=False,
                device=self.device,
                conf=self.confidence_threshold,
                stream=False,
            )
        except Exception as e:
            logger.warning("Detector inference failed on frame %d: %s", frame_sequence, e)
            return [], 0.0

        inference_ms = (time.monotonic() - t0) * 1000.0

        detections: List[Detection] = []
        if not results or results[0].boxes is None:
            return detections, inference_ms

        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            if cls_id not in self._all_vehicle_ids:
                continue
            if conf < self.confidence_threshold:
                continue

            xyxy = box.xyxy[0].tolist()
            bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
            vehicle_class = self._all_vehicle_ids[cls_id]

            det = Detection(
                vehicle_class=vehicle_class,
                confidence=conf,
                bbox=bbox,
                media_pts_ms=media_pts_ms,
                local_receive_monotonic=local_receive_monotonic,
                frame_sequence=frame_sequence,
            )
            if det.bbox_valid:
                detections.append(det)

        return detections, inference_ms
