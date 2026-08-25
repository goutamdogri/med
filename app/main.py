"""
FastAPI sidecar for MedCare ML Pipeline.

Provides HTTP endpoints for the Express backend to trigger daily rollover
and monthly retrain, and poll for run status. Runs inside the ML container
on port 8000.

Usage:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

app = FastAPI(title="MedCare ML Pipeline API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Run tracking ──────────────────────────────────────────────────────────────

@dataclass
class RunInfo:
    run_id: str
    status: str = "running"
    triggered_by: str = "api"
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    finished_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None
    steps_completed: list[str] = field(default_factory=list)

_runs: dict[str, RunInfo] = {}
_lock = Lock()


class RunResponse(BaseModel):
    run_id: str
    status: str
    message: str


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    triggered_by: str
    started_at: str
    finished_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None
    steps_completed: list[str] = []


# ─── Pipeline execution ────────────────────────────────────────────────────────

def _run_daily_pipeline(run_id: str) -> None:
    """Execute the full daily rollover pipeline."""
    info = _runs[run_id]
    t0 = time.time()
    try:
        r = subprocess.run(
            [PY, str(ROOT / "src" / "rolling_forecast.py"),
             "--full-chain", "--triggered-by", "api"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if r.returncode != 0:
            info.status = "failed"
            info.error = r.stderr[-500:] if r.stderr else f"exit code {r.returncode}"
        else:
            info.status = "completed"
            info.steps_completed = ["forecast", "replenishment", "allocation", "simulation", "alerts"]
    except subprocess.TimeoutExpired:
        info.status = "failed"
        info.error = "Pipeline timed out after 300s"
    except Exception as e:
        info.status = "failed"
        info.error = f"{type(e).__name__}: {e}"
    finally:
        info.duration_seconds = round(time.time() - t0, 1)
        info.finished_at = datetime.utcnow().isoformat()


def _run_retrain_pipeline(run_id: str) -> None:
    """Execute the full monthly retrain pipeline."""
    info = _runs[run_id]
    t0 = time.time()
    try:
        r = subprocess.run(
            [PY, str(ROOT / "src" / "retrain.py")],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=900,
        )
        if r.returncode != 0:
            info.status = "failed"
            info.error = r.stderr[-500:] if r.stderr else f"exit code {r.returncode}"
        else:
            info.status = "completed"
            info.steps_completed = [
                "train_lgbm", "torch_models", "ensemble",
                "replenishment", "allocation", "simulate", "alerts", "fill_derived"
            ]
    except subprocess.TimeoutExpired:
        info.status = "failed"
        info.error = "Pipeline timed out after 900s"
    except Exception as e:
        info.status = "failed"
        info.error = f"{type(e).__name__}: {e}"
    finally:
        info.duration_seconds = round(time.time() - t0, 1)
        info.finished_at = datetime.utcnow().isoformat()


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "medcare-ml-pipeline"}


@app.post("/run/daily", response_model=RunResponse)
def trigger_daily_rollover(triggered_by: str = "api"):
    """Trigger daily rollover forecast (LGBM + Chronos + full chain).
    
    Returns immediately with a run_id. Poll GET /run/{run_id}/status for completion.
    Typical duration: < 30 seconds.
    """
    with _lock:
        running = [r for r in _runs.values() if r.status == "running"]
        if running:
            raise HTTPException(
                status_code=409,
                detail=f"A run is already in progress: {running[0].run_id}"
            )
        run_id = str(uuid.uuid4())[:8]
        info = RunInfo(run_id=run_id, triggered_by=triggered_by)
        _runs[run_id] = info

    import threading
    t = threading.Thread(target=_run_daily_pipeline, args=(run_id,), daemon=True)
    t.start()

    return RunResponse(run_id=run_id, status="running", message="Daily rollover started")


@app.post("/run/retrain", response_model=RunResponse)
def trigger_retrain(triggered_by: str = "api"):
    """Trigger monthly full retrain (all models + full chain).
    
    Returns immediately with a run_id. Poll GET /run/{run_id}/status for completion.
    Typical duration: 3-5 minutes.
    """
    with _lock:
        running = [r for r in _runs.values() if r.status == "running"]
        if running:
            raise HTTPException(
                status_code=409,
                detail=f"A run is already in progress: {running[0].run_id}"
            )
        run_id = str(uuid.uuid4())[:8]
        info = RunInfo(run_id=run_id, triggered_by=triggered_by)
        _runs[run_id] = info

    import threading
    t = threading.Thread(target=_run_retrain_pipeline, args=(run_id,), daemon=True)
    t.start()

    return RunResponse(run_id=run_id, status="running", message="Monthly retrain started")


@app.get("/run/{run_id}/status", response_model=RunStatusResponse)
def get_run_status(run_id: str):
    """Poll run status. Returns 'running', 'completed', or 'failed'."""
    info = _runs.get(run_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return RunStatusResponse(
        run_id=info.run_id,
        status=info.status,
        triggered_by=info.triggered_by,
        started_at=info.started_at,
        finished_at=info.finished_at,
        duration_seconds=info.duration_seconds,
        error=info.error,
        steps_completed=info.steps_completed,
    )


@app.get("/runs")
def list_runs():
    """List all tracked runs (last 50)."""
    runs = sorted(_runs.values(), key=lambda r: r.started_at, reverse=True)[:50]
    return [
        {
            "run_id": r.run_id,
            "status": r.status,
            "triggered_by": r.triggered_by,
            "started_at": r.started_at,
            "finished_at": r.finished_at,
            "duration_seconds": r.duration_seconds,
        }
        for r in runs
    ]
