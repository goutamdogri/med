from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import CFG, PROCESSED, load_tables  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma4:e2b"


def build_alerts() -> list[dict]:
    tables = load_tables()
    repl = pd.read_parquet(PROCESSED / "replenishment_orders.parquet")
    writeoffs = pd.read_parquet(PROCESSED / "writeoff_risk.parquet")
    fcst = pd.read_parquet(PROCESSED / "forecasts_final.parquet")
    alerts = []

    crit_repl = repl[
        (repl["criticality"].isin(["critical", "high"]))
        & (repl["status"] != "ok")
    ].sort_values(["criticality", "days_of_supply_on_hand"])

    for _, r in crit_repl.iterrows():
        dos = r["days_of_supply_on_hand"]
        sev = "RED" if dos < CFG["review_policy"]["escalation"]["red_stock_days"] else "AMBER"
        alerts.append(
            {
                "severity": sev,
                "type": "shortage_risk",
                "sku_id": r["sku_id"],
                "region": r["region"],
                "facts": {
                    "criticality": r["criticality"],
                    "days_of_supply": None if np.isinf(dos) else round(float(dos), 1),
                    "lead_time_days": int(r["lead_time_days"]),
                    "recommended_order_units": int(r["order_qty"]),
                    "order_value_inr": int(r["order_value_inr"]),
                },
                "action": f"Expedite replenishment of {int(r['order_qty'])} units; consider transfer from metro DC.",
            }
        )

    if len(writeoffs):
        top_wo = writeoffs.head(10)
        for _, w in top_wo.iterrows():
            alerts.append(
                {
                    "severity": "AMBER",
                    "type": "expiry_writeoff_risk",
                    "sku_id": w["sku_id"],
                    "region": w["location"],
                    "facts": {
                        "units_at_risk": int(w["residual_writeoff_units"]),
                        "value_inr": int(w["residual_value_inr"]),
                        "days_to_expiry": int(w["days_to_expiry"]),
                    },
                    "action": "Apply markdown / donate-before-expiry program; no viable transfer destination found by optimizer.",
                }
            )

    surge = (
        fcst[fcst["horizon"] <= 7]
        .groupby(["region"])["sense_adjustment"]
        .mean()
        .sort_values(ascending=False)
    )
    trigger = CFG["review_policy"]["surge_trigger_sensed_uplift"]
    for region, adj in surge.items():
        if adj >= trigger:
            alerts.append(
                {
                    "severity": "RED" if adj > 0.3 else "AMBER",
                    "type": "demand_surge_detected",
                    "sku_id": "*",
                    "region": region,
                    "facts": {"sensed_uplift_pct": round(float(adj) * 100, 1)},
                    "action": f"Switch {region} to daily review cadence; pre-position stock ahead of flu curve.",
                }
            )

    return alerts


def review_cadence_recommendation(alerts: list[dict]) -> dict:
    surge_regions = {
        a["region"] for a in alerts if a["type"] == "demand_surge_detected"
    }
    red_count = sum(1 for a in alerts if a["severity"] == "RED")
    mode = "DAILY_SURGE_MODE" if surge_regions or red_count >= 5 else "WEEKLY_STANDARD"
    return {
        "mode": mode,
        "standard_cadence_days": CFG["review_policy"]["standard_cadence_days"],
        "surge_cadence_days": CFG["review_policy"]["surge_cadence_days"],
        "surge_regions": sorted(surge_regions),
        "red_alerts": red_count,
    }


def gemma_digest(alerts: list[dict], kpis: dict, cadence: dict) -> tuple[str, str]:
    """Returns (digest_text, model_used)."""
    facts = {
        "kpi_summary": kpis,
        "review_mode": cadence,
        "n_alerts": len(alerts),
        "top_alerts": [
            {k: v for k, v in a.items() if k in ("severity", "type", "sku_id", "region", "action")}
            for a in alerts[:12]
        ],
    }
    prompt = (
        "You are a pharma supply-chain planner. Write a concise daily escalation brief "
        "(max 180 words, plain text, no markdown headers) for the Chief Supply Chain Officer "
        "based strictly on these facts. State the review mode, the most urgent shortages, "
        "expiry risks, and 2-3 concrete actions.\n\nFACTS:\n"
        + json.dumps(facts, indent=1, default=str)
    )
    body = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a pharma supply-chain planner writing for the Chief Supply Chain Officer.",
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 1200},
        }
    ).encode()
    try:
        req = urllib.request.Request(
            OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            out = json.loads(resp.read())
        text = str(out.get("message", {}).get("content", "")).strip()
        if len(text) > 80:
            return text, OLLAMA_MODEL
    except Exception as e:
        print(f"gemma unavailable ({e}); using template digest")
    red = [a for a in alerts if a["severity"] == "RED"]
    lines = [
        f"DAILY ESCALATION BRIEF — mode: {cadence['mode']}",
        f"Red alerts: {len(red)}. Top shortage risks: "
        + "; ".join(f"{a['sku_id']} @ {a['region']}" for a in red[:5]),
        "Expiry program: execute markdown/donation for residual near-expiry batches.",
        "Cadence: weekly S&OP; escalate listed regions to daily review.",
    ]
    return "\n".join(lines), "template_fallback"


def main():
    kpi_df = pd.read_csv(PROCESSED / "kpi_summary.csv")
    proposed_row = kpi_df[kpi_df["policy"] == "proposed"].iloc[0].to_dict()
    sq_row = kpi_df[kpi_df["policy"] == "status_quo"].iloc[0].to_dict()
    kpis = {"proposed": proposed_row, "status_quo": sq_row}

    alerts = build_alerts()
    cadence = review_cadence_recommendation(alerts)
    digest, model_used = gemma_digest(alerts, kpis, cadence)

    payload = {"alerts": alerts, "review_cadence": cadence}
    (PROCESSED / "alerts.json").write_text(json.dumps(payload, indent=2, default=str))
    (PROCESSED / "alert_digest.md").write_text(digest)

    # dual-write: push alerts + AI digest to MySQL OUTPUT tables
    try:
        sys.path.insert(0, str(ROOT / "db"))
        import outputs as db_out

        db_out.write_alerts(
            alerts, digest_text=digest, review_mode=cadence["mode"],
            surge_regions=cadence["surge_regions"], red_count=cadence["red_alerts"],
            model_used=model_used, as_of_date=db_out.current_as_of(),
        )
    except Exception as exc:
        print(f"[alerts] MySQL write skipped: {type(exc).__name__}: {exc}")

    print(f"alerts: {len(alerts)} | mode: {cadence['mode']} | surge regions: {cadence['surge_regions']}")
    print("\n--- Gemma digest ---")
    print(digest)


if __name__ == "__main__":
    main()
