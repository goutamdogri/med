"""
runs.py — Persistent tracking of ML sidecar executions.

Run state (status, steps completed, error, timings) is stored in the
`pipeline_run` table instead of process memory so that a sidecar restart does
not lose in-flight or completed run history. The backend polls
GET /run/{run_id}/status which reads from here.

IMPORTANT: like every other table here, `pipeline_run` is written directly to
the database. The ML sidecar still never writes `pipeline_state`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from connection import get_engine, read_sql  # noqa: E402

RUN_TYPES = ("daily", "retrain")
STATUSES = ("running", "completed", "failed")


def _validate_type(run_type: str) -> None:
    if run_type not in RUN_TYPES:
        raise ValueError(f"run_type must be one of {RUN_TYPES}, got '{run_type}'")


def create_run(run_id: str, run_type: str, triggered_by: str = "api",
               as_of: str | None = None) -> None:
    """Insert a new 'running' run row. run_id is the PK."""
    _validate_type(run_type)
    with get_engine().begin() as conn:
        conn.execute(
            text("""
                INSERT INTO pipeline_run
                    (run_id, run_type, status, triggered_by, as_of)
                VALUES (:rid, :rtype, 'running', :trig, :asof)
            """),
            {"rid": run_id, "rtype": run_type, "trig": triggered_by, "asof": as_of},
        )


def finalize_run(run_id: str, status: str, duration_seconds: float,
                 steps_completed: list[str] | None = None,
                 error: str | None = None) -> None:
    """Mark a run completed/failed with its final timings and steps."""
    if status not in STATUSES or status == "running":
        raise ValueError(f"finalize status must be one of completed/failed, got '{status}'")
    with get_engine().begin() as conn:
        conn.execute(
            text("""
                UPDATE pipeline_run
                SET status = :st,
                    steps_completed = :steps,
                    error = :err,
                    duration_seconds = :dur,
                    finished_at = CURRENT_TIMESTAMP
                WHERE run_id = :rid
            """),
            {
                "rid": run_id,
                "st": status,
                "steps": ",".join(steps_completed or []),
                "err": error,
                "dur": round(float(duration_seconds), 1),
            },
        )


def get_run(run_id: str) -> dict | None:
    """Return a single run as a dict, or None if it does not exist."""
    df = read_sql(
        "SELECT run_id, run_type, status, triggered_by, as_of, steps_completed, "
        "       error, started_at, finished_at, duration_seconds "
        "FROM pipeline_run WHERE run_id = :rid",
        {"rid": run_id},
    )
    if df.empty:
        return None
    return _row_to_dict(df.iloc[0])


def list_runs(limit: int = 50) -> list[dict]:
    """Return the most recent runs, newest first."""
    df = read_sql(
        "SELECT run_id, run_type, status, triggered_by, as_of, steps_completed, "
        "       error, started_at, finished_at, duration_seconds "
        "FROM pipeline_run ORDER BY started_at DESC LIMIT :lim",
        {"lim": int(limit)},
    )
    return [_row_to_dict(row) for _, row in df.iterrows()]


# A run left in 'running' state beyond its own timeout is assumed to have been
# orphaned by a process crash. Such rows would otherwise block every new run
# forever, so they are not treated as genuinely in-progress.
_STALE_AFTER_S = {
    "daily": 600,    # > the 300s chain timeout + buffer
    "retrain": 1800,  # > the 900s retrain timeout + buffer
}


def running_run_id(run_type: str | None = None) -> str | None:
    """Return the run_id of a genuinely in-flight run of the given type (or any
    type when ``run_type`` is None), ignoring stale 'running' rows."""
    df = read_sql(
        "SELECT run_id, run_type, status, started_at "
        "FROM pipeline_run WHERE status = 'running' "
        "ORDER BY started_at ASC",
    )
    for _, row in df.iterrows():
        if run_type is not None and row["run_type"] != run_type:
            continue
        if row.get("started_at") is None or pd.isna(row["started_at"]):
            continue
        elapsed = (pd.Timestamp.now() - row["started_at"]).total_seconds()
        if elapsed < _STALE_AFTER_S.get(row["run_type"], 600):
            return row["run_id"]
    return None


def _row_to_dict(row: pd.Series) -> dict:
    return {
        "run_id": row["run_id"],
        "run_type": row["run_type"],
        "status": row["status"],
        "triggered_by": row["triggered_by"],
        "as_of": str(row["as_of"]) if pd.notna(row.get("as_of")) else None,
        "steps_completed": (
            [s for s in (row["steps_completed"] or "").split(",") if s]
            if pd.notna(row.get("steps_completed")) and row["steps_completed"]
            else []
        ),
        "error": row["error"] if pd.notna(row.get("error")) else None,
        "started_at": row["started_at"].isoformat() if pd.notna(row.get("started_at")) else None,
        "finished_at": row["finished_at"].isoformat() if pd.notna(row.get("finished_at")) else None,
        "duration_seconds": (
            float(row["duration_seconds"])
            if pd.notna(row.get("duration_seconds"))
            else None
        ),
    }
