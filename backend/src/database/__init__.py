"""Database package for CCTV Intelligence Platform (Step 12).

Provides:
- PostgreSQL + PostGIS connection and engine factory.
- Relational models for cameras, vehicles, observations, watchlists, matches, and journeys.
- Decoupled repository layer for spatial and relational operations.
"""
from src.database.connection import get_db, get_engine, init_engine

__all__ = ["get_engine", "init_engine", "get_db"]
