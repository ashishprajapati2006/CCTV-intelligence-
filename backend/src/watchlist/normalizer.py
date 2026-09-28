"""Watchlist plate normalization and validation utilities.

Ensures registration numbers are normalized identically to Step 6 (ANPR)
and Step 7 (Observed Vehicle DB).
"""
from __future__ import annotations

from src.ai.anpr.normalizer import normalize_raw_text
from src.ai.anpr.validator import validate_indian_plate


def normalize_watchlist_registration(raw_text: str) -> str:
    """Normalize a watchlist vehicle registration number.

    Strips whitespace, hyphens, non-alphanumeric characters, and leading IND badge.
    Converts to uppercase.
    """
    return normalize_raw_text(raw_text)


def validate_watchlist_registration(reg_number: str) -> bool:
    """Validate if registration conforms to recognized Indian plate standards."""
    norm = normalize_watchlist_registration(reg_number)
    is_valid, _, _ = validate_indian_plate(norm)
    return is_valid
