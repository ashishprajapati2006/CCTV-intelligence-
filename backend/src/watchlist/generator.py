"""Synthetic Watchlist Generator (Step 8).

Generates representative synthetic watchlist records:
  A. MATCHING records: Selects actual observed vehicles from Step 7 (vehicles.json).
  B. NON-MATCHING records: Generates realistic valid Indian registrations not in Step 7.

Ensures deterministic output when a random seed is provided.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.common.logging import get_logger
from src.watchlist.models import (
    ALLOWED_PRIORITIES,
    ALLOWED_VEHICLE_CATEGORIES,
    WatchlistVehicle,
)
from src.watchlist.normalizer import (
    normalize_watchlist_registration,
    validate_watchlist_registration,
)

logger = get_logger("watchlist_generator")

# Pool of valid Indian state/district prefixes and series letters for synthetic generation
SYNTHETIC_STATE_CODES = ["GJ", "MH", "DL", "RJ", "MP"]
SYNTHETIC_DISTRICTS = [f"{i:02d}" for i in range(1, 39)]
SYNTHETIC_SERIES = ["AA", "AB", "CD", "EF", "GH", "JK", "MN", "PQ", "RS", "XY", "ZZ"]


class SyntheticWatchlistGenerator:
    """Generates synthetic demonstration watchlist records with controlled matching/non-matching splits."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = random.Random(seed)

    def load_observed_registrations(self, observed_file_path: str) -> List[Dict[str, Any]]:
        """Load observed vehicle profiles from Step 7 vehicles.json."""
        p = Path(observed_file_path)
        if not p.exists():
            logger.warning("Observed vehicles file does not exist: %s", observed_file_path)
            return []

        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception as e:
            logger.error("Failed to read observed vehicles file: %s", e)
            return []

    def _generate_synthetic_nonmatch_plate(self, existing_regs: Set[str]) -> str:
        """Generate a valid synthetic Indian registration that does not collide with existing plates."""
        for _ in range(1000):
            state = self.rng.choice(SYNTHETIC_STATE_CODES)
            district = self.rng.choice(SYNTHETIC_DISTRICTS)
            series = self.rng.choice(SYNTHETIC_SERIES)
            num = self.rng.randint(1000, 9999)
            candidate = f"{state}{district}{series}{num}"

            norm = normalize_watchlist_registration(candidate)
            if norm not in existing_regs and validate_watchlist_registration(norm):
                return norm

        # Fallback if standard format collisions occur
        fallback = f"GJ99ZZ{self.rng.randint(1000, 9999)}"
        return fallback

    def generate_watchlist(
        self,
        observed_file_path: str,
        match_count: int = 5,
        nonmatch_count: int = 5,
    ) -> Tuple[List[WatchlistVehicle], Dict[str, Any]]:
        """Generate combined synthetic watchlist with matching and non-matching records.

        Args:
            observed_file_path: Path to Step 7 vehicles.json.
            match_count: Number of observed vehicles to include as MATCHING records.
            nonmatch_count: Number of non-matching synthetic vehicles to generate.

        Returns:
            Tuple of (list_of_watchlist_vehicles, generation_report_dict).
        """
        observed_records = self.load_observed_registrations(observed_file_path)
        observed_regs = [
            r.get("normalized_registration_number")
            for r in observed_records
            if r.get("normalized_registration_number")
        ]

        # Shuffle observed if seeded/randomized
        shuffled_observed = list(observed_regs)
        self.rng.shuffle(shuffled_observed)

        actual_match_count = min(match_count, len(shuffled_observed))
        selected_matches = shuffled_observed[:actual_match_count]

        watchlist: List[WatchlistVehicle] = []
        seen_normalized: Set[str] = set()
        wl_counter = 1

        # 1. Generate MATCHING Records
        categories = list(ALLOWED_VEHICLE_CATEGORIES)
        priorities = list(ALLOWED_PRIORITIES)

        for reg in selected_matches:
            norm = normalize_watchlist_registration(reg)
            if norm in seen_normalized:
                continue
            seen_normalized.add(norm)

            cat = self.rng.choice(categories)
            prio = self.rng.choice(priorities)
            w_id = f"WL-VEH-{wl_counter:06d}"
            wl_counter += 1

            watchlist.append(WatchlistVehicle(
                watchlist_id=w_id,
                registration_number=norm,
                normalized_registration_number=norm,
                category=cat,
                priority=prio,
                status="ACTIVE",
                source="SYNTHETIC_DEMO",
                synthetic=True,
                description="Synthetic demonstration record (Target: Step 7 Observed Match)",
                notes="Configured to match actual observed vehicle in CCTV network",
            ))

        # 2. Generate NON-MATCHING Records
        all_excluded_regs = set(observed_regs).union(seen_normalized)
        generated_nonmatches: List[str] = []

        for _ in range(nonmatch_count):
            nonmatch_reg = self._generate_synthetic_nonmatch_plate(all_excluded_regs)
            seen_normalized.add(nonmatch_reg)
            all_excluded_regs.add(nonmatch_reg)
            generated_nonmatches.append(nonmatch_reg)

            cat = self.rng.choice(categories)
            prio = self.rng.choice(priorities)
            # Include 1 inactive demo case if multiple nonmatches
            status = "INACTIVE" if len(generated_nonmatches) == 1 and nonmatch_count > 3 else "ACTIVE"
            w_id = f"WL-VEH-{wl_counter:06d}"
            wl_counter += 1

            watchlist.append(WatchlistVehicle(
                watchlist_id=w_id,
                registration_number=nonmatch_reg,
                normalized_registration_number=nonmatch_reg,
                category=cat,
                priority=prio,
                status=status,
                source="SYNTHETIC_DEMO",
                synthetic=True,
                description="Synthetic demonstration record (Target: Non-Matching Control)",
                notes="Configured to demonstrate negative lookup case (unobserved vehicle)",
            ))

        report = {
            "observed_vehicles_available": len(observed_regs),
            "synthetic_matching_records": len(selected_matches),
            "synthetic_nonmatching_records": len(generated_nonmatches),
            "total_watchlist_records": len(watchlist),
            "expected_match_coverage": f"{len(selected_matches)} / {len(observed_regs)} observed vehicles",
            "seed_used": self.seed,
        }

        return watchlist, report
