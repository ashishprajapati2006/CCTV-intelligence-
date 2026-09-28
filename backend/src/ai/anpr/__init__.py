"""Step 6 ANPR/OCR & Multi-Frame Consensus Package."""

from src.ai.anpr.schemas import (
    OCRResult,
    NormalizedOCRResult,
    TextCorrectionLog,
    ANPRObservation,
    TrackANPRConsensus,
)
from src.ai.anpr.ocr import (
    OCRRecognizer,
    EasyOCREngine,
    OpenCVContourOCREngine,
    OCREngineFacade,
)
from src.ai.anpr.normalizer import (
    normalize_raw_text,
    normalize_with_audit,
)
from src.ai.anpr.validator import (
    validate_indian_plate,
    INDIAN_STATE_CODES,
)
from src.ai.anpr.scoring import (
    compute_observation_score,
    ObservationScoreWeights,
)
from src.ai.anpr.consensus import (
    MultiFrameConsensusEngine,
    _levenshtein_distance,
)
from src.ai.anpr.eligibility import (
    CandidateEligibilityEvaluator,
    EligibilityResult,
)
from src.ai.anpr.recognizer import ANPRRecognizer

__all__ = [
    "OCRResult",
    "NormalizedOCRResult",
    "TextCorrectionLog",
    "ANPRObservation",
    "TrackANPRConsensus",
    "OCRRecognizer",
    "EasyOCREngine",
    "OpenCVContourOCREngine",
    "OCREngineFacade",
    "normalize_raw_text",
    "normalize_with_audit",
    "validate_indian_plate",
    "INDIAN_STATE_CODES",
    "compute_observation_score",
    "ObservationScoreWeights",
    "MultiFrameConsensusEngine",
    "_levenshtein_distance",
    "CandidateEligibilityEvaluator",
    "EligibilityResult",
    "ANPRRecognizer",
]
