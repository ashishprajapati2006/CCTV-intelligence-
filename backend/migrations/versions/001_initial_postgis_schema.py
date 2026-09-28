"""001 Initial PostgreSQL PostGIS Schema

Revision ID: 001_initial_postgis_schema
Revises: None
Create Date: 2026-09-22 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text
from src.database.connection import Base
import src.database.models

# revision identifiers, used by Alembic.
revision: str = "001_initial_postgis_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))

    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
