"""Text normalization and character correction audit tracking for ANPR.

RULES:
  1. Uppercase & strip non-alphanumeric characters (spaces, hyphens, dots).
  2. Strip leading 'IND' country badge prefix if detected on HSRP plates.
  3. Preserves raw OCR text untouched.
  4. Audits all character corrections (original_char, corrected_char, position, reason).
  5. Contextual character swap (e.g. O -> 0 in numeric positions, 0 -> O, 6 -> G in state code alpha positions).
"""
from __future__ import annotations

import re
from typing import List
from src.ai.anpr.schemas import NormalizedOCRResult, TextCorrectionLog
from src.common.logging import get_logger

logger = get_logger("anpr_normalizer")

# Regex to strip non-alphanumeric characters
_CLEAN_RE = re.compile(r"[^A-Z0-9]")

# Common OCR confusion mappings
_ALPHA_TO_NUMERIC = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "Q": "0", "L": "4", "A": "4", "D": "0", "E": "6"}
_NUMERIC_TO_ALPHA = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "4": "A"}


def normalize_raw_text(text: str) -> str:
    """Basic normalization: uppercase and strip non-alphanumeric characters."""
    if not text:
        return ""
    upper = text.upper()
    cleaned = _CLEAN_RE.sub("", upper)
    # Strip leading IND badge if present
    if cleaned.startswith("IND") and len(cleaned) >= 11:
        cleaned = cleaned[3:]
    return cleaned


def normalize_with_audit(
    raw_text: str,
    apply_contextual_corrections: bool = True,
) -> NormalizedOCRResult:
    """Normalize raw OCR text and produce an audited NormalizedOCRResult.

    Contextual correction rules for standard Indian 10-char format (e.g. GJ01AB1234):
      - Strip leading 'IND' if present on HSRP plates
      - Pos 0, 1: State Code (Alpha) -> convert digits to alpha if obvious confusion (e.g. 6 -> G)
      - Pos 2, 3: District Code (Numeric) -> convert alpha to digits if obvious confusion
      - Pos 4, 5: Series Code (Alpha) -> convert digits to alpha if length == 10
      - Pos 6-9: Serial Number (Numeric) -> convert alpha to digits
    """
    clean_text = _CLEAN_RE.sub("", raw_text.upper()) if raw_text else ""
    if not clean_text:
        return NormalizedOCRResult(
            original_text=raw_text or "",
            normalized_text="",
            corrections=[],
            is_valid_format=False,
        )

    corrections: List[TextCorrectionLog] = []

    # Strip leading 'IND' country badge if present
    if clean_text.startswith("IND") and len(clean_text) >= 11:
        corrections.append(TextCorrectionLog(
            original_char="IND",
            corrected_char="",
            position=0,
            reason="strip_ind_country_badge_prefix"
        ))
        clean_text = clean_text[3:]

    final_chars = list(clean_text)

    # Apply contextual corrections if string length is typical (9 or 10 chars)
    if apply_contextual_corrections and len(clean_text) in (9, 10):
        length = len(clean_text)
        is_10 = (length == 10)

        for i, char in enumerate(clean_text):
            # Pos 0, 1: Must be State Code (Alpha)
            if i in (0, 1) and char in _NUMERIC_TO_ALPHA:
                corr = _NUMERIC_TO_ALPHA[char]
                final_chars[i] = corr
                corrections.append(TextCorrectionLog(
                    original_char=char,
                    corrected_char=corr,
                    position=i,
                    reason=f"state_code_alpha_position_{char}_to_{corr}"
                ))

            # Pos 2, 3: Must be District Code (Numeric)
            elif i in (2, 3) and char in _ALPHA_TO_NUMERIC:
                corr = _ALPHA_TO_NUMERIC[char]
                final_chars[i] = corr
                corrections.append(TextCorrectionLog(
                    original_char=char,
                    corrected_char=corr,
                    position=i,
                    reason=f"district_code_numeric_position_{char}_to_{corr}"
                ))

            # Pos 4, 5 (for 10-char format): Series Code (Alpha)
            elif is_10 and i in (4, 5) and char in _NUMERIC_TO_ALPHA:
                corr = _NUMERIC_TO_ALPHA[char]
                final_chars[i] = corr
                corrections.append(TextCorrectionLog(
                    original_char=char,
                    corrected_char=corr,
                    position=i,
                    reason=f"series_code_alpha_position_{char}_to_{corr}"
                ))

            # Tail characters: Serial Number (Numeric)
            elif (is_10 and i >= 6) or (not is_10 and i >= 5):
                if char in _ALPHA_TO_NUMERIC:
                    corr = _ALPHA_TO_NUMERIC[char]
                    final_chars[i] = corr
                    corrections.append(TextCorrectionLog(
                        original_char=char,
                        corrected_char=corr,
                        position=i,
                        reason=f"serial_num_numeric_position_{char}_to_{corr}"
                    ))

    normalized_final = "".join(final_chars)
    is_valid = bool(re.match(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$", normalized_final))

    return NormalizedOCRResult(
        original_text=raw_text,
        normalized_text=normalized_final,
        corrections=corrections,
        is_valid_format=is_valid,
    )
