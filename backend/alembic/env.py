from logging.config import fileConfig
import sys
from pathlib import Path

from sqlalchemy import pool
from sqlalchemy.engine import create_engine

from alembic import context


# ---------------------------------------------------------------------
# Project Path
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))


# ---------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------

from app.core.config import settings
from app.database.base import Base
from app.models.prediction import Prediction

# ---------------------------------------------------------------------
# Alembic Configuration
# ---------------------------------------------------------------------

config = context.config


# ---------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


# ---------------------------------------------------------------------
# SQLAlchemy Metadata
# ---------------------------------------------------------------------

target_metadata = Base.metadata


# ---------------------------------------------------------------------
# Database URL
# ---------------------------------------------------------------------

database_url = settings.DATABASE_URL


# ---------------------------------------------------------------------
# Offline Migration
# ---------------------------------------------------------------------

def run_migrations_offline() -> None:
    """Run migrations without creating a database connection."""

    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={
            "paramstyle": "named"
        },
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------
# Online Migration
# ---------------------------------------------------------------------

def run_migrations_online() -> None:
    """Run migrations using a live database connection."""

    connectable = create_engine(
        database_url,
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


# ---------------------------------------------------------------------
# Run Migration
# ---------------------------------------------------------------------

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()