"""Query service layer for Representative Synthetic Watchlist (Step 8).

Provides high-level lookup, filtering, and validation functions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.watchlist.models import WatchlistVehicle
from src.watchlist.normalizer import normalize_watchlist_registration
from src.watchlist.repository import JSONWatchlistRepository, WatchlistRepository
from src.watchlist.validator import WatchlistValidator


class WatchlistQueries:
    """High-level query service over WatchlistRepository."""

    def __init__(self, repository: Optional[WatchlistRepository] = None):
        self.repo = repository or JSONWatchlistRepository()
        self.validator = WatchlistValidator()

    def get_watchlist_vehicle(self, registration_number: str) -> Optional[WatchlistVehicle]:
        """Lookup a vehicle in the watchlist by registration number (normalized)."""
        return self.repo.get_vehicle(registration_number)

    def get_active_watchlist(self) -> List[WatchlistVehicle]:
        """Fetch all ACTIVE watchlist records."""
        return self.repo.get_active_vehicles()

    def list_watchlist_by_category(self, category: str) -> List[WatchlistVehicle]:
        """List watchlist records by category (e.g. DEMO_STOLEN_VEHICLE)."""
        return self.repo.list_by_category(category)

    def list_watchlist_by_priority(self, priority: str) -> List[WatchlistVehicle]:
        """List watchlist records by priority (e.g. HIGH, CRITICAL)."""
        return self.repo.list_by_priority(priority)

    def all_vehicles(self) -> List[WatchlistVehicle]:
        """Return all watchlist vehicle records."""
        return self.repo.all_vehicles()

    def count_watchlist(self) -> int:
        """Total records in the watchlist."""
        return self.repo.count()

    def validate_watchlist(self, observed_file_path: Optional[str] = None) -> Dict[str, Any]:
        """Validate all records in repository against format rules and Step 7 observed vehicles."""
        observed_regs: Set[str] = set()
        if observed_file_path and Path(observed_file_path).exists():
            try:
                with open(observed_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for r in data:
                            if r.get("normalized_registration_number"):
                                observed_regs.add(r["normalized_registration_number"])
            except Exception:
                pass

        records = self.repo.all_vehicles()
        return self.validator.validate_dataset(records, observed_registrations=observed_regs)
