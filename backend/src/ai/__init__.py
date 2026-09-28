"""AI package for CCTV vehicle detection, tracking, frame quality scoring, and license plate region detection.

Steps 4 & 5 unified exports.
"""
from src.ai.schemas import (
    Detection,
    VehicleTrack,
    DetectionEvent,
    FrameCandidate,
    VehicleFrameCandidate,
    PlateCandidate,
)
from src.ai.detector import VehicleDetector
from src.ai.tracker import VehicleTracker
from src.ai.frame_selector import (
    evaluate_frame,
    evaluate_frame_extended,
    update_best_frames,
    select_top_vehicle_frames,
    save_best_frames,
    compute_sharpness,
    compute_brightness,
    compute_contrast,
    compute_truncation,
    score_frame,
    score_frame_extended,
    FrameQualityWeights,
)
from src.ai.plate_detector import PlateDetector, PlateDetection
from src.ai.plate_quality import validate_plate_bbox, score_plate, PlateQualityMetrics
from src.ai.plate_preprocessor import (
    crop_plate_with_margin,
    generate_preprocessing_variants,
    save_plate_crops,
    warp_perspective,
)
from src.ai.candidate_manager import CandidateManager

__all__ = [
    # Schemas
    "Detection",
    "VehicleTrack",
    "DetectionEvent",
    "FrameCandidate",
    "VehicleFrameCandidate",
    "PlateCandidate",
    # Step 4 AI
    "VehicleDetector",
    "VehicleTracker",
    "evaluate_frame",
    "update_best_frames",
    "save_best_frames",
    # Step 5 AI
    "evaluate_frame_extended",
    "select_top_vehicle_frames",
    "compute_sharpness",
    "compute_brightness",
    "compute_contrast",
    "compute_truncation",
    "score_frame",
    "score_frame_extended",
    "FrameQualityWeights",
    "PlateDetector",
    "PlateDetection",
    "validate_plate_bbox",
    "score_plate",
    "PlateQualityMetrics",
    "crop_plate_with_margin",
    "generate_preprocessing_variants",
    "save_plate_crops",
    "warp_perspective",
    "CandidateManager",
]
