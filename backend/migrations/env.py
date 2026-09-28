"""Alembic Environment Configuration (Step 12).

Connects Alembic to the SQLAlchemy Base metadata and handles:
- Dynamic connection string from environment variables
- PostGIS extension preservation
- Migration execution in both offline and online modes
"""
from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool, text

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database.connection import Base, get_database_url
import src.database.models  # Ensure all models are registered

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url():
    return get_database_url()


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = get_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        # Enable PostGIS extension if Postgres
        if connectable.dialect.name == "postgresql":
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
            connection.commit()

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
