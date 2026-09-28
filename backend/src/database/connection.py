"""Database Connection & Session Factory for CCTV Intelligence Platform (Step 12).

Manages:
- Environment variable configuration (DATABASE_HOST, DATABASE_PORT, DATABASE_NAME, DATABASE_USER, DATABASE_PASSWORD).
- Credential masking in logs.
- Engine creation with connection pooling.
- Session factory and context manager.
- PostgreSQL / PostGIS extension detection and health checks.
"""
from __future__ import annotations

import os
import re
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, scoped_session, sessionmaker

from src.common.logging import get_logger

logger = get_logger("database.connection")


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy declarative models."""
    pass


def mask_db_url(url: str) -> str:
    """Mask password from database URL for secure logging."""
    if not url:
        return ""
    return re.sub(r":([^:@]+)@", r":****@", str(url))


def get_database_url() -> str:
    """Build database connection URL from environment variables."""
    # Explicit DATABASE_URL takes priority if set
    explicit_url = os.getenv("DATABASE_URL")
    if explicit_url:
        return explicit_url

    host = os.getenv("DATABASE_HOST", "localhost")
    port = os.getenv("DATABASE_PORT", "5432")
    dbname = os.getenv("DATABASE_NAME", "cctv_platform")
    user = os.getenv("DATABASE_USER", "cctv_user")
    password = os.getenv("DATABASE_PASSWORD", "")

    if password:
        encoded_password = quote_plus(password)
        return f"postgresql+psycopg2://{user}:{encoded_password}@{host}:{port}/{dbname}"
    else:
        return f"postgresql+psycopg2://{user}@{host}:{port}/{dbname}"


# Global singletons
_engine: Optional[Engine] = None
_SessionFactory: Optional[sessionmaker] = None


def init_engine(db_url: Optional[str] = None, echo: bool = False, **kwargs) -> Engine:
    """Initialize the global SQLAlchemy engine."""
    global _engine, _SessionFactory
    url = db_url or get_database_url()

    masked = mask_db_url(url)
    logger.info("Initializing database engine: %s", masked)

    if url.startswith("sqlite"):
        engine = create_engine(
            url,
            echo=echo,
            connect_args={"check_same_thread": False},
            **kwargs,
        )
    else:
        pool_size = int(os.getenv("DATABASE_POOL_SIZE", "10"))
        max_overflow = int(os.getenv("DATABASE_MAX_OVERFLOW", "20"))
        engine = create_engine(
            url,
            echo=echo,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            **kwargs,
        )

    _engine = engine
    _SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _engine


def get_engine() -> Engine:
    """Get or lazily initialize the database engine."""
    global _engine
    if _engine is None:
        _engine = init_engine()
    return _engine


def get_session_factory() -> sessionmaker:
    """Get the sessionmaker factory."""
    global _SessionFactory
    if _SessionFactory is None:
        get_engine()
    return _SessionFactory  # type: ignore


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Context manager yielding a transactional database session."""
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error("Database transaction rolled back due to error: %s", e)
        raise
    finally:
        session.close()


def check_postgis_status(engine: Optional[Engine] = None) -> Dict[str, Any]:
    """Check database connectivity and PostGIS extension status.

    Returns:
        Dict with keys: connected, postgres_version, postgis_version, postgis_installed, is_sqlite
    """
    eng = engine or get_engine()
    status: Dict[str, Any] = {
        "connected": False,
        "is_sqlite": eng.dialect.name == "sqlite",
        "postgres_version": None,
        "postgis_installed": False,
        "postgis_version": None,
        "error": None,
    }

    try:
        with eng.connect() as conn:
            status["connected"] = True
            if status["is_sqlite"]:
                # SQLite dialect
                res = conn.execute(text("SELECT sqlite_version()")).scalar()
                status["postgres_version"] = f"SQLite {res}"
                status["postgis_installed"] = False
                status["postgis_version"] = "N/A (SQLite)"
            else:
                # PostgreSQL
                pg_ver = conn.execute(text("SELECT version()")).scalar()
                status["postgres_version"] = str(pg_ver).split(",")[0] if pg_ver else "Unknown"

                # Check PostGIS extension
                ext_check = conn.execute(
                    text("SELECT extname, extversion FROM pg_extension WHERE extname = 'postgis'")
                ).fetchone()

                if ext_check:
                    status["postgis_installed"] = True
                    status["postgis_version"] = ext_check[1]
                else:
                    status["postgis_installed"] = False
                    status["postgis_version"] = None
    except Exception as e:
        status["connected"] = False
        status["error"] = str(e)
        logger.warning("Database connectivity / PostGIS check failed: %s", e)

    return status
