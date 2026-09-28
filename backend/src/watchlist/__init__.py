"""Representative Synthetic Watchlist & Real-Time Matching Package (Steps 8 & 9).

Provides:
  - Step 8: Synthetic demonstration watchlists.
  - Step 9: Real-time ANPR watchlist matching, eligibility gating, deduplication, and decision auditing.
"""
from src.watchlist.models import (
    ALLOWED_MATCH_DECISIONS,
    ALLOWED_PRIORITIES,
    ALLOWED_STATUSES,
    ALLOWED_VEHICLE_CATEGORIES,
    MatchDecision,
    WatchlistMetadataSnapshot,
    WatchlistPerson,
    WatchlistVehicle,
)
from src.watchlist.normalizer import (
    normalize_watchlist_registration,
    validate_watchlist_registration,
)
from src.watchlist.repository import (
    JSONWatchlistRepository,
    WatchlistRepository,
)
from src.watchlist.generator import SyntheticWatchlistGenerator
from src.watchlist.validator import WatchlistValidator
from src.watchlist.queries import WatchlistQueries
from src.watchlist.eligibility import (
    MatchEligibilityConfig,
    MatchEligibilityEvaluator,
)
from src.watchlist.matcher import WatchlistMatcher
from src.watchlist.deduplication import AlertDeduplicator
from src.watchlist.decision import MatchDecisionEngine

__all__ = [
    "ALLOWED_MATCH_DECISIONS",
    "ALLOWED_PRIORITIES",
    "ALLOWED_STATUSES",
    "ALLOWED_VEHICLE_CATEGORIES",
    "MatchDecision",
    "WatchlistMetadataSnapshot",
    "WatchlistPerson",
    "WatchlistVehicle",
    "normalize_watchlist_registration",
    "validate_watchlist_registration",
    "WatchlistRepository",
    "JSONWatchlistRepository",
    "SyntheticWatchlistGenerator",
    "WatchlistValidator",
    "WatchlistQueries",
    "MatchEligibilityConfig",
    "MatchEligibilityEvaluator",
    "WatchlistMatcher",
    "AlertDeduplicator",
    "MatchDecisionEngine",
]
