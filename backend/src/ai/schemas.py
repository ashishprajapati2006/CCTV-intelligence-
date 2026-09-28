"""Shared data schemas for Steps 1-5.

PTS TIMING RULE (applies to every field using time):
  - media_pts_ms            : FROM the video container (cap.get(CAP_PROP_POS_MSEC)).
                              This is the AUTHORITATIVE video timeline.
  - local_receive_monotonic : FROM time.monotonic() on frame arrival.
                              Used ONLY for pipeline performance diagnostics.
  - NEVER derive video timing from frame_number / fps or datetime.now().

Step 5 boundary:
  PlateCandidate answers "Where is the plate region?"
  It does NOT contain a registration_number field.
  That belongs to Step 6 (ANPR/OCR).
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


# ════════════════════════════════════════════════════════════════════════════
#  Step 4 schemas (unchanged)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Detection:
    """A single raw detection from the vehicle detector on one frame."""
    vehicle_class: str           # "car", "truck", "bus", "motorcycle", ...
    confidence: float            # 0.0 - 1.0
    bbox: List[int]              # [x1, y1, x2, y2] in pixel coords
    media_pts_ms: Optional[float] = None
    local_receive_monotonic: Optional[float] = None
    frame_sequence: int = 0

    @property
    def bbox_area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def bbox_valid(self) -> bool:
        if len(self.bbox) != 4:
            return False
        x1, y1, x2, y2 = self.bbox
        return x2 > x1 and y2 > y1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FrameCandidate:
    """A candidate snapshot frame collected for a track (Step 4 / Step 5 input)."""
    frame_sequence: int
    media_pts_ms: Optional[float]
    local_receive_monotonic: float
    sharpness: float         # Laplacian variance (higher = sharper)
    bbox_area: int           # Larger = more detail
    brightness: float        # Mean pixel luminance
    quality_score: float     # Composite ranking score
    file_path: Optional[str] = None  # Set after saving


@dataclass
class VehicleTrack:
    """Live state of one tracked vehicle across consecutive frames.

    Timing:
      first_seen_pts_ms / last_seen_pts_ms use CONTAINER PTS.
      They must never be derived from frame counts or FPS.
    """
    track_id: str
    camera_id: str
    vehicle_class: str
    first_seen_pts_ms: Optional[float]
    last_seen_pts_ms: Optional[float]
    frame_count: int = 0
    missed_frames: int = 0
    current_bbox: List[int] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    best_frames: List[FrameCandidate] = field(default_factory=list)
    is_active: bool = True

    @property
    def mean_confidence(self) -> float:
        return float(sum(self.confidences) / len(self.confidences)) if self.confidences else 0.0

    @property
    def best_frame_paths(self) -> List[str]:
        return [fc.file_path for fc in self.best_frames if fc.file_path]

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "camera_id": self.camera_id,
            "vehicle_class": self.vehicle_class,
            "first_seen_pts_ms": self.first_seen_pts_ms,
            "last_seen_pts_ms": self.last_seen_pts_ms,
            "frame_count": self.frame_count,
            "mean_confidence": round(self.mean_confidence, 4),
            "last_bbox": self.current_bbox,
            "is_active": self.is_active,
            "best_frame_paths": self.best_frame_paths,
        }


@dataclass
class DetectionEvent:
    """One persisted detection observation (written to JSONL)."""
    camera_id: str
    track_id: str
    pts_ms: Optional[float]
    vehicle_class: str
    confidence: float
    bbox: List[int]
    frame_sequence: int
    local_receive_monotonic: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "pts_ms": self.pts_ms,
            "vehicle_class": self.vehicle_class,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
            "frame_sequence": self.frame_sequence,
            "local_receive_monotonic": self.local_receive_monotonic,
        }


# ════════════════════════════════════════════════════════════════════════════
#  Step 5 schemas (new)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class VehicleFrameCandidate:
    """An enhanced vehicle frame candidate produced by Step 5 quality scoring.

    This extends the Step 4 FrameCandidate concept with richer metrics and
    full tracking provenance.

    Timing: pts_ms is the authoritative video timeline (container PTS).
    """
    camera_id: str
    track_id: str
    frame_id: int                          # == frame_sequence
    pts_ms: Optional[float]               # container PTS; None if unavailable
    has_valid_pts: bool                   # explicit flag rather than implicit None check
    local_receive_monotonic: float

    # Vehicle detection info
    bbox: List[int]                        # vehicle bbox [x1, y1, x2, y2]
    vehicle_class: str
    detection_confidence: float

    # Quality metrics (all normalized to [0, 1])
    sharpness_score: float                 # Laplacian variance, normalised
    brightness_score: float                # proximity to ideal luminance
    contrast_score: float                  # RMS contrast, normalised
    area_score: float                      # bbox area, normalised
    truncation_score: float                # 1.0 = no truncation, 0.0 = fully clipped
    vehicle_quality_score: float           # composite weighted score

    image_path: Optional[str] = None       # path to saved full-frame snapshot

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "frame_id": self.frame_id,
            "pts_ms": self.pts_ms,
            "has_valid_pts": self.has_valid_pts,
            "local_receive_monotonic": round(self.local_receive_monotonic, 6),
            "bbox": self.bbox,
            "vehicle_class": self.vehicle_class,
            "detection_confidence": round(self.detection_confidence, 4),
            "sharpness_score": round(self.sharpness_score, 4),
            "brightness_score": round(self.brightness_score, 4),
            "contrast_score": round(self.contrast_score, 4),
            "area_score": round(self.area_score, 4),
            "truncation_score": round(self.truncation_score, 4),
            "vehicle_quality_score": round(self.vehicle_quality_score, 4),
            "image_path": self.image_path,
        }


@dataclass
class PlateCandidate:
    """A detected license plate region for one vehicle frame.

    SCOPE: This schema answers "Where is the plate?" only.
    It does NOT contain a registration_number field.
    Registration number determination belongs to Step 6 (ANPR/OCR).

    Coordinates:
      plate_bbox is expressed in FULL-FRAME pixel coordinates, even when
      detection was performed on a vehicle-ROI crop.
    """
    camera_id: str
    track_id: str
    frame_id: int
    pts_ms: Optional[float]               # container PTS; None if unavailable
    has_valid_pts: bool
    local_receive_monotonic: float

    # Detection
    plate_bbox: List[int]                  # [x1, y1, x2, y2] in FULL-FRAME coords
    plate_confidence: float                # detector confidence 0-1
    detector_version: str                  # e.g. "haar_v1", "yolo_lp_v1"

    # Plate quality (separate from detector confidence)
    plate_quality_score: float             # composite [0, 1]
    plate_width: int
    plate_height: int
    sharpness_score: float
    brightness_score: float
    contrast_score: float

    # File paths (populated after saving)
    original_crop_path: Optional[str] = None
    processed_crop_paths: Dict[str, str] = field(default_factory=dict)

    # Rejection metadata (None if candidate is accepted)
    rejection_reason: Optional[str] = None

    @property
    def is_accepted(self) -> bool:
        return self.rejection_reason is None

    @property
    def combined_score(self) -> float:
        """Ranking score used for candidate deduplication and top-N selection."""
        return 0.5 * self.plate_quality_score + 0.5 * self.plate_confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "frame_id": self.frame_id,
            "pts_ms": self.pts_ms,
            "has_valid_pts": self.has_valid_pts,
            "local_receive_monotonic": round(self.local_receive_monotonic, 6),
            "plate_bbox": self.plate_bbox,
            "plate_confidence": round(self.plate_confidence, 4),
            "detector_version": self.detector_version,
            "plate_quality_score": round(self.plate_quality_score, 4),
            "plate_width": self.plate_width,
            "plate_height": self.plate_height,
            "sharpness_score": round(self.sharpness_score, 4),
            "brightness_score": round(self.brightness_score, 4),
            "contrast_score": round(self.contrast_score, 4),
            "combined_score": round(self.combined_score, 4),
            "original_crop_path": self.original_crop_path,
            "processed_crop_paths": self.processed_crop_paths,
            "rejection_reason": self.rejection_reason,
            # STEP 6 BOUNDARY: registration_number is intentionally absent.
        }
