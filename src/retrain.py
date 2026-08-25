from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
PY = sys.executable

# Monthly retrain trains on whatever demand_history currently holds in MySQL
# (backend-fed). build_dataset.py is only needed to regenerate the synthetic
# demo dataset from the raw Kaggle CSV — use --rebuild-data for that.
STEPS = [
    ("train_lgbm", [PY, str(ROOT / "src" / "train_lgbm.py")]),
    ("torch_models", [PY, str(ROOT / "src" / "torch_models.py")]),
    ("ensemble", [PY, str(ROOT / "src" / "ensemble.py")]),
    ("replenishment", [PY, str(ROOT / "src" / "replenishment.py")]),
    ("allocation", [PY, str(ROOT / "src" / "allocation.py")]),
    ("simulate", [PY, str(ROOT / "src" / "simulate.py")]),
    ("alerts", [PY, str(ROOT / "src" / "alerts.py")]),
]

BUILD_STEP = ("build_dataset", [PY, str(ROOT / "src" / "build_dataset.py")])


def set_as_of_to_latest_demand() -> str | None:
    """Point config.yaml at the newest ingested day so every stage shares one origin."""
    sys.path.insert(0, str(ROOT / "db"))
    from connection import scalar  # noqa: E402

    try:
        latest = scalar("SELECT MAX(date) FROM demand_history")
    except Exception:
        return None
    if latest is None:
        return None
    cfg_path = ROOT / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    prev = cfg["project"]["as_of_date"]
    cfg["project"]["as_of_date"] = str(pd_ts(latest))
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"[retrain] as-of {prev} -> {cfg['project']['as_of_date']}")
    return prev


def pd_ts(d) -> str:
    import pandas as pd

    return pd.Timestamp(d).date().isoformat()


def run_pipeline(skip_torch: bool = False, rebuild_data: bool = False) -> dict:
    log = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}, "ok": True}
    try:
        if not rebuild_data:
            set_as_of_to_latest_demand()
        steps = ([BUILD_STEP] if rebuild_data else []) + STEPS
        for name, cmd in steps:
            if skip_torch and name == "torch_models":
                log["steps"][name] = {"status": "skipped"}
                print(f"[retrain] SKIP {name} (existing forecasts reused if fresh)")
                continue
            t0 = time.time()
            print(f"\n[retrain] >>> {name} ...", flush=True)
            r = subprocess.run(cmd, cwd=ROOT)
            dt = round(time.time() - t0, 1)
            if r.returncode != 0:
                log["steps"][name] = {"status": "failed", "secs": dt}
                log["ok"] = False
                raise RuntimeError(f"{name} failed after {dt}s")
            log["steps"][name] = {"status": "done", "secs": dt}
        # refresh [DERIVED] tables from the accumulated MySQL data
        t0 = time.time()
        r = subprocess.run([PY, str(ROOT / "db" / "fill_derived.py")], cwd=ROOT)
        dt = round(time.time() - t0, 1)
        name = "fill_derived"
        if r.returncode != 0:
            log["steps"][name] = {"status": "failed", "secs": dt}
            log["ok"] = False
            raise RuntimeError(f"{name} failed after {dt}s")
        log["steps"][name] = {"status": "done", "secs": dt}
    except Exception as exc:
        print(f"\n[retrain] FAILED: {exc}")
        log["ok"] = False
    else:
        log["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")

    out = ROOT / "models" / "retrain_log.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(log, indent=2))
    print(f"\n[retrain] {'OK' if log['ok'] else 'FAILED'} -> {out}")
    return log


def main():
    ap = argparse.ArgumentParser(description="Monthly model retrain (PostgreSQL-backed)")
    ap.add_argument("--skip-torch", action="store_true", help="reuse existing NN forecasts")
    ap.add_argument("--rebuild-data", action="store_true",
                    help="regenerate synthetic dataset from raw CSV first (demo reset)")
    args = ap.parse_args()
    log = run_pipeline(skip_torch=args.skip_torch, rebuild_data=args.rebuild_data)
    sys.exit(0 if log["ok"] else 1)


if __name__ == "__main__":
    main()
