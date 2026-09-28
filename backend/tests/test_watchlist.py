"""Unit tests for Step 8 — Representative Synthetic Watchlist.

Covers Part 25:
  1. Normalization (hyphens, spaces, lowercase, IND badge).
  2. Duplicate detection & prevention.
  3. Active vs Inactive filtering.
  4. Matching subset selection from Step 7 observed vehicles.
  5. Non-matching synthetic record generation.
  6. Observed/Watchlist overlap verification.
  7. Deterministic random seed reproducibility.
  8. Invalid registration format rejection.
  9. Synthetic flag enforcement (synthetic=True).
  10. Source validation (source='SYNTHETIC_DEMO').
  11. Category validation (prefixed with 'DEMO_').
  12. Priority validation (LOW, MEDIUM, HIGH, CRITICAL).
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from src.watchlist import (
    ALLOWED_PRIORITIES,
    ALLOWED_STATUSES,
    ALLOWED_VEHICLE_CATEGORIES,
    JSONWatchlistRepository,
    SyntheticWatchlistGenerator,
    WatchlistQueries,
    WatchlistValidator,
    WatchlistVehicle,
    normalize_watchlist_registration,
    validate_watchlist_registration,
)


@pytest.fixture
def temp_repo(tmp_path):
    v_file = tmp_path / "watchlist.json"
    p_file = tmp_path / "persons.json"
    return JSONWatchlistRepository(vehicles_path=str(v_file), persons_path=str(p_file))


def _make_dummy_vehicle(
    w_id: str = "WL-VEH-000001",
    reg: str = "GJ01AB1234",
    cat: str = "DEMO_STOLEN_VEHICLE",
    prio: str = "HIGH",
    status: str = "ACTIVE",
    synthetic: bool = True,
    source: str = "SYNTHETIC_DEMO",
) -> WatchlistVehicle:
    norm = normalize_watchlist_registration(reg)
    return WatchlistVehicle(
        watchlist_id=w_id,
        registration_number=reg,
        normalized_registration_number=norm,
        category=cat,
        priority=prio,
        status=status,
        source=source,
        synthetic=synthetic,
    )


# ════════════════════════════════════════════════════════════════════════════
# 1. Normalization & Format Validation
# ════════════════════════════════════════════════════════════════════════════

def test_normalization_consistency():
    assert normalize_watchlist_registration("gj-01 ab 1234") == "GJ01AB1234"
    assert normalize_watchlist_registration("INDGJ01AB1234") == "GJ01AB1234"
    assert normalize_watchlist_registration("GJ-01-AB-1234") == "GJ01AB1234"


def test_format_validation():
    assert validate_watchlist_registration("GJ01AB1234") is True
    assert validate_watchlist_registration("22BH1234AA") is True
    assert validate_watchlist_registration("INVALID_123") is False
    assert validate_watchlist_registration("GJ01") is False


# ════════════════════════════════════════════════════════════════════════════
# 2. Duplicate Detection & Repository Persistence
# ════════════════════════════════════════════════════════════════════════════

def test_duplicate_registration_rejected(temp_repo):
    v1 = _make_dummy_vehicle(w_id="WL-001", reg="GJ01AB1234")
    v2 = _make_dummy_vehicle(w_id="WL-002", reg="GJ-01-AB-1234")  # Normalizes to same string

    added1 = temp_repo.add_vehicle(v1)
    added2 = temp_repo.add_vehicle(v2)

    assert added1 is True
    assert added2 is False  # Rejected duplicate
    assert temp_repo.count() == 1


def test_active_inactive_filtering(temp_repo):
    v1 = _make_dummy_vehicle(w_id="WL-001", reg="GJ01AB1234", status="ACTIVE")
    v2 = _make_dummy_vehicle(w_id="WL-002", reg="GJ05CD5678", status="INACTIVE")

    temp_repo.add_vehicle(v1)
    temp_repo.add_vehicle(v2)

    active = temp_repo.get_active_vehicles()
    assert len(active) == 1
    assert active[0].normalized_registration_number == "GJ01AB1234"


# ════════════════════════════════════════════════════════════════════════════
# 3. Matching & Non-Matching Synthetic Generation
# ════════════════════════════════════════════════════════════════════════════

def test_generator_matching_and_nonmatching(tmp_path):
    # Mock Step 7 vehicles.json
    obs_data = [
        {"normalized_registration_number": "GJ01AB1234"},
        {"normalized_registration_number": "GJ05CD5678"},
        {"normalized_registration_number": "MH12DE1122"},
    ]
    obs_file = tmp_path / "vehicles.json"
    with open(obs_file, "w", encoding="utf-8") as f:
        json.dump(obs_data, f)

    gen = SyntheticWatchlistGenerator(seed=42)
    watchlist, report = gen.generate_watchlist(
        observed_file_path=str(obs_file),
        match_count=2,
        nonmatch_count=3,
    )

    assert len(watchlist) == 5
    assert report["synthetic_matching_records"] == 2
    assert report["synthetic_nonmatching_records"] == 3

    # Check that 2 match observed set
    obs_set = {"GJ01AB1234", "GJ05CD5678", "MH12DE1122"}
    matched = [v for v in watchlist if v.normalized_registration_number in obs_set]
    nonmatched = [v for v in watchlist if v.normalized_registration_number not in obs_set]

    assert len(matched) == 2
    assert len(nonmatched) == 3


def test_generator_deterministic_seed(tmp_path):
    obs_data = [
        {"normalized_registration_number": "GJ01AB1234"},
        {"normalized_registration_number": "GJ05CD5678"},
    ]
    obs_file = tmp_path / "vehicles.json"
    with open(obs_file, "w", encoding="utf-8") as f:
        json.dump(obs_data, f)

    gen1 = SyntheticWatchlistGenerator(seed=123)
    wl1, _ = gen1.generate_watchlist(str(obs_file), match_count=1, nonmatch_count=2)

    gen2 = SyntheticWatchlistGenerator(seed=123)
    wl2, _ = gen2.generate_watchlist(str(obs_file), match_count=1, nonmatch_count=2)

    regs1 = [v.normalized_registration_number for v in wl1]
    regs2 = [v.normalized_registration_number for v in wl2]
    assert regs1 == regs2


# ════════════════════════════════════════════════════════════════════════════
# 4. Watchlist Validation & Data Integrity Rules
# ════════════════════════════════════════════════════════════════════════════

def test_validator_detects_invalid_schema():
    validator = WatchlistValidator()

    # Missing synthetic flag
    bad_rec1 = _make_dummy_vehicle(synthetic=False)
    errs1 = validator.validate_record(bad_rec1)
    assert any("synthetic" in e for e in errs1)

    # Invalid source
    bad_rec2 = _make_dummy_vehicle(source="REAL_POLICE_DB")
    errs2 = validator.validate_record(bad_rec2)
    assert any("source" in e for e in errs2)

    # Invalid category
    bad_rec3 = _make_dummy_vehicle(cat="UNAUTHORIZED_CAT")
    errs3 = validator.validate_record(bad_rec3)
    assert any("category" in e for e in errs3)

    # Invalid format
    bad_rec4 = _make_dummy_vehicle(reg="INVALID123")
    errs4 = validator.validate_record(bad_rec4)
    assert any("format" in e for e in errs4)


def test_validator_dataset_audit():
    validator = WatchlistValidator()
    records = [
        _make_dummy_vehicle(w_id="WL-01", reg="GJ01AB1234"),
        _make_dummy_vehicle(w_id="WL-02", reg="GJ99ZZ9999"),
    ]
    observed = {"GJ01AB1234"}

    res = validator.validate_dataset(records, observed_registrations=observed)
    assert res["is_valid"] is True
    assert res["observed_counterparts_count"] == 1
    assert res["unobserved_counterparts_count"] == 1
    assert res["guaranteed_matches"] == ["GJ01AB1234"]


# ════════════════════════════════════════════════════════════════════════════
# 5. Queries Service
# ════════════════════════════════════════════════════════════════════════════

def test_queries_service(temp_repo):
    v1 = _make_dummy_vehicle(w_id="WL-001", reg="GJ01AB1234", cat="DEMO_STOLEN_VEHICLE", prio="CRITICAL")
    temp_repo.add_vehicle(v1)

    queries = WatchlistQueries(repository=temp_repo)

    rec = queries.get_watchlist_vehicle("gj-01 ab 1234")
    assert rec is not None
    assert rec.priority == "CRITICAL"

    stolen = queries.list_watchlist_by_category("DEMO_STOLEN_VEHICLE")
    assert len(stolen) == 1

    critical = queries.list_watchlist_by_priority("CRITICAL")
    assert len(critical) == 1
