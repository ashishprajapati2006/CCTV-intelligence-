"""Vehicle Registration Number Format Validator for Step 6.

Supports two modes:
  - 'indian'   : Validate against Indian RTO registration patterns (default for camera footage)
  - 'european' : Validate against common European/International plate patterns (for synthetic data)
  - 'auto'     : Try both Indian and European formats

INDIAN PATTERNS:
  1. Standard 10-char : State (2) + District (2) + Series (2-3) + Number (4) e.g. GJ01AB1234
  2. Legacy / Short   : State (2) + District (1-2) + Series (1) + Number (1-4) e.g. GJ05A1234
  3. Bharat Series BH : Year (2) + BH + Number (4) + Series (1-2) e.g. 22BH1234AA

EUROPEAN PATTERNS (covers most EU/UK/US/generic foreign footage):
  1. UK New Style : AB12 CDE -> AB12CDE  (2 letters + 2 digits + 3 letters)
  2. UK Old Style : ABC 123D -> ABC123D  (3 letters + 3 digits + 1 letter)
  3. EU Generic   : AB1234 / ABC1234 / 1234AB / 1234ABC (alpha-numeric 5-9 chars)
  4. US Style     : ABC-1234 / 1234-ABC (3-7 chars, mixed)
  5. Generic LP   : Any clean alphanumeric 4-12 chars (permissive for unknown countries)

CRITICAL SAFETY RULE:
  Does NOT fabricate missing characters or hallucinate text to force regex matches.
"""
from __future__ import annotations

import re
from typing import Tuple, Optional
from src.common.logging import get_logger

logger = get_logger("anpr_validator")

# ── Indian State & UT codes ────────────────────────────────────────────────────
INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN",
    "GA", "GJ", "HR", "HP", "JK", "JH", "KA", "KL", "LA", "LD",
    "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "OR", "PB", "PY",
    "RJ", "SK", "TN", "TS", "TR", "UP", "UK", "UA", "WB"
}

# ── Indian Regex Patterns ──────────────────────────────────────────────────────
_RE_IN_STANDARD = re.compile(r"^([A-Z]{2})([0-9]{2})([A-Z]{2,3})([0-9]{4})$")
_RE_IN_LEGACY   = re.compile(r"^([A-Z]{2})([0-9]{1,2})([A-Z]{1})([0-9]{1,4})$")
_RE_IN_BHARAT   = re.compile(r"^([0-9]{2})BH([0-9]{4})([A-Z]{1,2})$")

# ── European / International Regex Patterns ────────────────────────────────────
# UK new style: AB12CDE
_RE_EU_UK_NEW = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$")
# UK old style: ABC123D or ABC1234
_RE_EU_UK_OLD = re.compile(r"^[A-Z]{3}[0-9]{3,4}[A-Z]?$")
# EU Generic: starts with letters ends with digits, or vice versa, 5-9 chars
_RE_EU_GENERIC_1 = re.compile(r"^[A-Z]{1,4}[0-9]{2,5}$")         # e.g. AB1234, ABC123
_RE_EU_GENERIC_2 = re.compile(r"^[0-9]{2,4}[A-Z]{1,4}$")         # e.g. 1234AB, 123ABC
_RE_EU_GENERIC_3 = re.compile(r"^[A-Z]{1,3}[0-9]{1,4}[A-Z]{1,3}$")  # e.g. AB123CD
# US style: 3-7 chars mixed alphanumeric
_RE_EU_US = re.compile(r"^[A-Z0-9]{3,8}$")
# Generic permissive: any 4-12 char alphanumeric (catches most world plates)
_RE_EU_PERMISSIVE = re.compile(r"^[A-Z0-9]{4,12}$")


def validate_indian_plate(text: str) -> Tuple[bool, Optional[str], float]:
    """Validate against Indian RTO registration patterns.

    Returns:
        (is_valid, format_name, format_confidence_boost)
    """
    if not text or len(text) < 6:
        return False, "incomplete_text", 0.0

    m_bh = _RE_IN_BHARAT.match(text)
    if m_bh:
        return True, "bharat_series_bh", 1.0

    m_std = _RE_IN_STANDARD.match(text)
    if m_std:
        state_code = m_std.group(1)
        if state_code in INDIAN_STATE_CODES:
            return True, "standard_indian_10", 1.0
        return True, "standard_indian_unrecognized_state", 0.80

    m_leg = _RE_IN_LEGACY.match(text)
    if m_leg:
        state_code = m_leg.group(1)
        if state_code in INDIAN_STATE_CODES:
            return True, "short_indian_legacy", 0.85
        return True, "short_indian_unrecognized_state", 0.70

    return False, "unrecognized_format", 0.0


def validate_european_plate(text: str) -> Tuple[bool, Optional[str], float]:
    """Validate against European / International plate patterns.

    Uses a tiered approach from most specific to most permissive.

    Returns:
        (is_valid, format_name, format_confidence_boost)
    """
    if not text or len(text) < 4:
        return False, "incomplete_text", 0.0

    if _RE_EU_UK_NEW.match(text):
        return True, "uk_new_style", 1.0

    if _RE_EU_UK_OLD.match(text):
        return True, "uk_old_style", 0.95

    if _RE_EU_GENERIC_1.match(text):
        return True, "eu_generic_alpha_numeric", 0.85

    if _RE_EU_GENERIC_2.match(text):
        return True, "eu_generic_numeric_alpha", 0.85

    if _RE_EU_GENERIC_3.match(text):
        return True, "eu_generic_sandwich", 0.80

    if _RE_EU_US.match(text):
        return True, "us_style", 0.75

    if _RE_EU_PERMISSIVE.match(text):
        return True, "eu_permissive_generic", 0.65

    return False, "unrecognized_format", 0.0


def validate_plate(text: str, plate_format: str = "indian") -> Tuple[bool, Optional[str], float]:
    """Unified validator that delegates based on plate_format.

    Args:
        text: Uppercase, clean alphanumeric plate string.
        plate_format: One of 'indian', 'european', 'auto'.

    Returns:
        (is_valid, format_name, format_confidence_boost)
    """
    text = text.upper().strip()
    # Remove common OCR artifacts: spaces, hyphens
    text = re.sub(r"[\s\-]", "", text)

    if plate_format == "indian":
        return validate_indian_plate(text)
    elif plate_format == "european":
        return validate_european_plate(text)
    elif plate_format == "auto":
        # Try Indian first, then European
        result = validate_indian_plate(text)
        if result[0]:
            return result
        return validate_european_plate(text)
    else:
        logger.warning("Unknown plate_format '%s', defaulting to indian", plate_format)
        return validate_indian_plate(text)
