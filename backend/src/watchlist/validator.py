"""Watchlist Validation Engine (Step 8).

Validates watchlist records for:
  - Schema conformity (synthetic=True, source='SYNTHETIC_DEMO').
  - Valid Indian registration format.
  - Absence of duplicate normalized registration numbers.
  - Category and priority validity.
  - Overlap against Step 7 Observed Vehicle Database.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from src.watchlist.models import (
    ALLOWED_PRIORITIES,
    ALLOWED_STATUSES,
    ALLOWED_VEHICLE_CATEGORIES,
    WatchlistVehicle,
)
from src.watchlist.normalizer import (
    normalize_watchlist_registration,
    validate_watchlist_registration,
)


class WatchlistValidator:
    """Audits and validates watchlist datasets against data integrity rules and Step 7 observations."""

    @staticmethod
    def validate_record(record: WatchlistVehicle) -> List[str]:
        """Validate a single WatchlistVehicle record. Returns list of errors (empty if valid)."""
        errors = []

        if not record.synthetic:
            errors.append("Field 'synthetic' must be True for all demonstration records")
        if record.source != "SYNTHETIC_DEMO":
            errors.append(f"Field 'source' must be 'SYNTHETIC_DEMO', got: {record.source}")
        if record.category not in ALLOWED_VEHICLE_CATEGORIES:
            errors.append(f"Invalid category '{record.category}'. Allowed: {ALLOWED_VEHICLE_CATEGORIES}")
        if record.priority not in ALLOWED_PRIORITIES:
            errors.append(f"Invalid priority '{record.priority}'. Allowed: {ALLOWED_PRIORITIES}")
        if record.status not in ALLOWED_STATUSES:
            errors.append(f"Invalid status '{record.status}'. Allowed: {ALLOWED_STATUSES}")

        # Check normalization
        norm = normalize_watchlist_registration(record.registration_number)
        if norm != record.normalized_registration_number:
            errors.append(
                f"Normalized registration mismatch: '{record.normalized_registration_number}' != '{norm}'"
            )

        # Check format validity
        if not validate_watchlist_registration(norm):
            errors.append(f"Registration '{norm}' does not conform to valid Indian plate format")

        return errors

    def validate_dataset(
        self,
        watchlist_records: List[WatchlistVehicle],
        observed_registrations: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Audit an entire watchlist dataset and cross-reference with observed registrations."""
        observed_set = observed_registrations or set()

        total_records = len(watchlist_records)
        seen_normalized: Set[str] = set()
        duplicate_records: List[str] = []
        invalid_format_records: List[str] = []
        schema_errors: Dict[str, List[str]] = {}
        active_records = 0
        inactive_records = 0

        matching_records: List[str] = []
        nonmatching_records: List[str] = []

        for v in watchlist_records:
            norm = v.normalized_registration_number
            # Check duplicates
            if norm in seen_normalized:
                duplicate_records.append(norm)
            else:
                seen_normalized.add(norm)

            # Check individual errors
            errs = self.validate_record(v)
            if errs:
                schema_errors[v.watchlist_id] = errs
                if any("format" in e for e in errs):
                    invalid_format_records.append(norm)

            if v.status == "ACTIVE":
                active_records += 1
            else:
                inactive_records += 1

            # Cross-reference with Step 7 observed database
            if norm in observed_set:
                matching_records.append(norm)
            else:
                nonmatching_records.append(norm)

        is_valid = (
            len(duplicate_records) == 0
            and len(schema_errors) == 0
            and len(invalid_format_records) == 0
        )

        return {
            "is_valid": is_valid,
            "total_records": total_records,
            "active_records": active_records,
            "inactive_records": inactive_records,
            "duplicate_count": len(duplicate_records),
            "duplicates": duplicate_records,
            "invalid_format_count": len(invalid_format_records),
            "invalid_formats": invalid_format_records,
            "schema_errors_count": len(schema_errors),
            "schema_errors": schema_errors,
            "observed_counterparts_count": len(matching_records),
            "guaranteed_matches": matching_records,
            "unobserved_counterparts_count": len(nonmatching_records),
            "non_matching_records": nonmatching_records,
            "observed_coverage_ratio": f"{len(matching_records)} / {len(observed_set)} observed vehicles",
        }
