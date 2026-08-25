"""
pipeline_state.py — Read/write the simulated "today" date.

The pipeline_state table holds a single row with simulated_today.
All ML pipelines read from here instead of MAX(date) or config.yaml.
The backend advances this date; the ML project only reads it.
"""

from __future__ import annotations

import pandas as pd

from connection import scalar, get_engine
from sqlalchemy import text


def get_simulated_today() -> pd.Timestamp:
    """Read the current simulated date from DB."""
    val = scalar("SELECT simulated_today FROM pipeline_state WHERE id = 1")
    if val is None:
        raise RuntimeError(
            "pipeline_state table is empty. Run db/seed_staging.py first."
        )
    return pd.Timestamp(val)


def set_simulated_today(date: pd.Timestamp) -> None:
    """Update the simulated date in DB."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE pipeline_state SET simulated_today = :d, "
                "updated_at = CURRENT_TIMESTAMP WHERE id = 1"
            ),
            {"d": date.date()},
        )


def advance_simulated_today(days: int = 1) -> pd.Timestamp:
    """Advance by N days, return new date."""
    current = get_simulated_today()
    new_date = current + pd.Timedelta(days=days)
    set_simulated_today(new_date)
    return new_date
