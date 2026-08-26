"""Shared PostgreSQL access layer for the medcare pipeline (SQLAlchemy + psycopg2).

Connection URL resolution order:
  1. PHARMA_DB_URL environment variable
  2. KEY=VALUE pairs in <repo>/.env
  3. built-in local default
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_URL = "postgresql://postgres:1828@localhost:5432/medcare"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


_load_dotenv(ROOT / ".env")
DB_URL = os.environ.get("PHARMA_DB_URL", DEFAULT_DB_URL)
DB_NAME = urlparse(DB_URL).path.lstrip("/").split("?")[0] or "pharma_sc"

_cached_engine: Engine | None = None


def get_engine() -> Engine:
    global _cached_engine
    if _cached_engine is None:
        _cached_engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    return _cached_engine


def read_sql(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame."""
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def insert_df(df: pd.DataFrame, table: str, chunk: int = 1000) -> int:
    """Append rows to a table (auto-resets identity sequence if needed)."""
    if df.empty:
        print(f"  = {table:<32} nothing to insert")
        return 0
    # Reset identity sequence to avoid collisions with seed data
    with get_engine().begin() as conn:
        conn.execute(text(
            f"SELECT setval(pg_get_serial_sequence(:t, 'id'), "
            f"COALESCE((SELECT MAX(id) FROM {table}), 1))"
        ), {"t": table})
    df.to_sql(table, con=get_engine(), if_exists="append", index=False,
              method="multi", chunksize=chunk)
    print(f"  + {table:<32} {len(df):>7} rows inserted")
    return len(df)


def delete_run(engine: Engine, table: str, as_of_date) -> None:
    """Idempotency helper: remove any previous rows for this run date."""
    with engine.begin() as conn:
        conn.execute(
            text(f"DELETE FROM {table} WHERE as_of_date = :d"), {"d": as_of_date}
        )


def write_run(df: pd.DataFrame, table: str, as_of_date, chunk: int = 1000) -> int:
    """Write all rows of one run: delete previous rows of that run, then insert."""
    engine = get_engine()
    delete_run(engine, table, as_of_date)
    return insert_df(df, table, chunk=chunk)


def scalar(sql: str, params: dict | None = None):
    with get_engine().connect() as conn:
        row = conn.execute(text(sql), params or {}).first()
    return None if row is None else row[0]
