from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
PY = sys.executable

STEPS = [
    ("build_dataset", [PY, str(ROOT / "src" / "build_dataset.py")]),
    ("train_lgbm", [PY, str(ROOT / "src" / "train_lgbm.py")]),
    ("torch_models", [PY, str(ROOT / "src" / "torch_models.py")]),
    ("ensemble", [PY, str(ROOT / "src" / "ensemble.py")]),
    ("replenishment", [PY, str(ROOT / "src" / "replenishment.py")]),
    ("allocation", [PY, str(ROOT / "src" / "allocation.py")]),
    ("simulate", [PY, str(ROOT / "src" / "simulate.py")]),
    ("alerts", [PY, str(ROOT / "src" / "alerts.py")]),
]


def run_pipeline(skip_torch: bool = False) -> dict:
    backup = PROCESSED.parent / f"processed_backup_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copytree(PROCESSED, backup)

    log = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}, "ok": True}
    try:
        for name, cmd in STEPS:
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
    except Exception as exc:
        print(f"\n[retrain] FAILED: {exc}")
        print(f"[retrain] rolling back processed artifacts from {backup.name}")
        shutil.rmtree(PROCESSED)
        shutil.move(str(backup), str(PROCESSED))
        log["rollback"] = True
    else:
        shutil.rmtree(backup, ignore_errors=True)
        log["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")

    out = PROCESSED / "retrain_log.json"
    out.write_text(json.dumps(log, indent=2))
    print(f"\n[retrain] {'OK' if log['ok'] else 'FAILED'} -> {out}")
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-torch", action="store_true", help="reuse existing NN forecasts")
    args = ap.parse_args()
    log = run_pipeline(skip_torch=args.skip_torch)
    sys.exit(0 if log["ok"] else 1)


if __name__ == "__main__":
    main()
