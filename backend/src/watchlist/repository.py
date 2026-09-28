"""Repository layer for Representative Synthetic Watchlist (Step 8).

Defines:
  1. WatchlistRepository: Abstract base class ready for future PostgreSQL swap.
  2. JSONWatchlistRepository: Concrete JSON storage implementation.

Storage Layout:
  data/watchlist/
    vehicles/
      watchlist.json
      watchlist.schema.json
    persons/
      watchlist.json
    metadata/
      README.md
"""
from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.common.logging import get_logger
from src.watchlist.models import WatchlistPerson, WatchlistVehicle
from src.watchlist.normalizer import normalize_watchlist_registration

logger = get_logger("watchlist_repo")


class WatchlistRepository(abc.ABC):
    """Abstract interface for watchlist persistence and querying."""

    @abc.abstractmethod
    def add_vehicle(self, vehicle: WatchlistVehicle) -> bool:
        """Add vehicle to watchlist. Returns False if duplicate normalized registration exists."""
        pass

    @abc.abstractmethod
    def get_vehicle(self, registration_number: str) -> Optional[WatchlistVehicle]:
        """Fetch watchlist record by registration number (normalized lookup)."""
        pass

    @abc.abstractmethod
    def get_active_vehicles(self) -> List[WatchlistVehicle]:
        """Fetch all ACTIVE watchlist vehicles eligible for matching."""
        pass

    @abc.abstractmethod
    def list_by_category(self, category: str) -> List[WatchlistVehicle]:
        """Filter watchlist vehicles by category."""
        pass

    @abc.abstractmethod
    def list_by_priority(self, priority: str) -> List[WatchlistVehicle]:
        """Filter watchlist vehicles by priority."""
        pass

    @abc.abstractmethod
    def all_vehicles(self) -> List[WatchlistVehicle]:
        """Return all watchlist vehicles."""
        pass

    @abc.abstractmethod
    def count(self) -> int:
        """Total count of watchlist records."""
        pass

    @abc.abstractmethod
    def save_all(self, vehicles: List[WatchlistVehicle]) -> None:
        """Overwrite or save full list of watchlist vehicles atomically."""
        pass


class JSONWatchlistRepository(WatchlistRepository):
    """Concrete file-based JSON repository for watchlist data."""

    def __init__(
        self,
        vehicles_path: str = "data/watchlist/vehicles/watchlist.json",
        persons_path: str = "data/watchlist/persons/watchlist.json",
    ):
        self.vehicles_file = Path(vehicles_path)
        self.persons_file = Path(persons_path)
        self.vehicles_file.parent.mkdir(parents=True, exist_ok=True)
        self.persons_file.parent.mkdir(parents=True, exist_ok=True)

        self._vehicles: Dict[str, WatchlistVehicle] = {}  # norm_reg -> WatchlistVehicle
        self._persons: Dict[str, WatchlistPerson] = {}    # watchlist_id -> WatchlistPerson
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load watchlist records from disk into in-memory index."""
        if self.vehicles_file.exists():
            try:
                with open(self.vehicles_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        v = WatchlistVehicle.from_dict(item)
                        self._vehicles[v.normalized_registration_number] = v
            except Exception as e:
                logger.error("Failed to load vehicle watchlist from %s: %s", self.vehicles_file, e)

        if self.persons_file.exists():
            try:
                with open(self.persons_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        p = WatchlistPerson.from_dict(item)
                        self._persons[p.watchlist_id] = p
            except Exception as e:
                logger.error("Failed to load person watchlist from %s: %s", self.persons_file, e)

    def add_vehicle(self, vehicle: WatchlistVehicle) -> bool:
        """Add vehicle. Rejects if normalized registration already exists (enforces uniqueness)."""
        norm = normalize_watchlist_registration(vehicle.normalized_registration_number)
        if norm in self._vehicles:
            logger.warning("Duplicate registration rejected: %s", norm)
            return False

        self._vehicles[norm] = vehicle
        self._flush_vehicles()
        return True

    def get_vehicle(self, registration_number: str) -> Optional[WatchlistVehicle]:
        norm = normalize_watchlist_registration(registration_number)
        return self._vehicles.get(norm)

    def get_active_vehicles(self) -> List[WatchlistVehicle]:
        return [v for v in self._vehicles.values() if v.status == "ACTIVE"]

    def list_by_category(self, category: str) -> List[WatchlistVehicle]:
        return [v for v in self._vehicles.values() if v.category == category]

    def list_by_priority(self, priority: str) -> List[WatchlistVehicle]:
        return [v for v in self._vehicles.values() if v.priority == priority]

    def all_vehicles(self) -> List[WatchlistVehicle]:
        return list(self._vehicles.values())

    def count(self) -> int:
        return len(self._vehicles)

    def save_all(self, vehicles: List[WatchlistVehicle]) -> None:
        """Atomically overwrite vehicle watchlist."""
        self._vehicles = {
            v.normalized_registration_number: v
            for v in vehicles
        }
        self._flush_vehicles()

    def _flush_vehicles(self) -> None:
        """Write all vehicle records atomically to disk."""
        tmp_file = self.vehicles_file.with_suffix(".tmp")
        payload = [v.to_dict() for v in self._vehicles.values()]
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp_file.replace(self.vehicles_file)
