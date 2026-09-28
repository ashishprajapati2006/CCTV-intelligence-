"""Shared data schemas for Step 6 ANPR/OCR & Multi-Frame Consensus.

PTS TIMING RULE (enforced throughout):
  - pts_ms : FROM the video container (cap.get(CAP_PROP_POS_MSEC)).
             This is the AUTHORITATIVE video timeline.
  - local_receive_monotonic : FROM time.monotonic() on frame arrival.
             Used ONLY for pipeline performance diagnostics.
  - NEVER derive video timing from frame_number / fps or datetime.now().

RECOGNITION STATES:
  - CONFIRMED : High confidence, multi-frame agreement, valid Indian plate format.
  - PROBABLE  : High confidence single-frame or candidate with minor ambiguity.
  - UNCERTAIN : Conflicting OCR readings or weak evidence support.
  - UNREADABLE: No valid/readable text extracted from any candidate frame.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class OCRResult:
    """Raw output from a single OCR engine run on a plate crop."""
    raw_text: str
    confidence: float                              # 0.0 – 1.0 engine confidence
    character_confidences: Optional[List[float]]  # None if engine doesn't provide char-level
    processing_time_ms: float
    engine_name: str
    engine_version: str
    variant_name: str                              # e.g., "original", "upscaled", "contrast"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "confidence": round(self.confidence, 4),
            "character_confidences": [round(c, 4) for c in self.character_confidences] if self.character_confidences else None,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "engine_name": self.engine_name,
            "engine_version": self.engine_version,
            "variant_name": self.variant_name,
        }


@dataclass
class TextCorrectionLog:
    """Audit entry for a character correction during text normalization."""
    original_char: str
    corrected_char: str
    position: int
    reason: str                                     # e.g., "numeric_position_O_to_0", "alpha_position_0_to_O"


@dataclass
class NormalizedOCRResult:
    """Normalized OCR reading with audit log of character corrections."""
    original_text: str
    normalized_text: str
    corrections: List[TextCorrectionLog] = field(default_factory=list)
    is_valid_format: bool = False
    format_name: Optional[str] = None              # e.g., "standard_indian_10", "old_indian_9"

    @property
    def has_corrections(self) -> bool:
        return len(self.corrections) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "corrections": [asdict(c) for c in self.corrections],
            "is_valid_format": self.is_valid_format,
            "format_name": self.format_name,
        }


@dataclass
class ANPRObservation:
    """A single ANPR observation linking a Step 5 plate candidate to an OCR reading."""
    camera_id: str
    track_id: str
    frame_id: int
    pts_ms: Optional[float]
    has_valid_pts: bool
    local_receive_monotonic: float

    plate_bbox: List[int]
    plate_detection_confidence: float
    plate_quality_score: float

    variant_name: str
    raw_ocr: OCRResult
    normalized_ocr: NormalizedOCRResult
    observation_score: float                       # composite evidence score for this observation

    image_crop_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "frame_id": self.frame_id,
            "pts_ms": self.pts_ms,
            "has_valid_pts": self.has_valid_pts,
            "local_receive_monotonic": round(self.local_receive_monotonic, 6),
            "plate_bbox": self.plate_bbox,
            "plate_detection_confidence": round(self.plate_detection_confidence, 4),
            "plate_quality_score": round(self.plate_quality_score, 4),
            "variant_name": self.variant_name,
            "raw_ocr": self.raw_ocr.to_dict(),
            "normalized_ocr": self.normalized_ocr.to_dict(),
            "observation_score": round(self.observation_score, 4),
            "image_crop_path": self.image_crop_path,
        }


@dataclass
class TrackANPRConsensus:
    """Final track-level ANPR result produced by multi-frame consensus."""
    camera_id: str
    track_id: str
    final_registration_number: Optional[str]        # None if UNREADABLE or UNCERTAIN without quorum
    recognition_status: str                         # "CONFIRMED", "PROBABLE", "UNCERTAIN", "UNREADABLE"
    consensus_score: float                          # [0.0 – 1.0] confidence score

    supporting_frame_count: int                    # number of distinct video frames supporting this number
    total_frames_examined: int
    total_observations_examined: int

    first_seen_pts_ms: Optional[float]
    last_seen_pts_ms: Optional[float]

    supporting_frame_ids: List[int] = field(default_factory=list)
    candidate_scores: Dict[str, float] = field(default_factory=dict) # {string -> weighted_score}

    evidence_chain: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "final_registration_number": self.final_registration_number,
            "recognition_status": self.recognition_status,
            "consensus_score": round(self.consensus_score, 4),
            "supporting_frame_count": self.supporting_frame_count,
            "total_frames_examined": self.total_frames_examined,
            "total_observations_examined": self.total_observations_examined,
            "first_seen_pts_ms": self.first_seen_pts_ms,
            "last_seen_pts_ms": self.last_seen_pts_ms,
            "supporting_frame_ids": self.supporting_frame_ids,
            "candidate_scores": {k: round(v, 4) for k, v in self.candidate_scores.items()},
            "evidence_chain": self.evidence_chain,
        }
