from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import CFG, PROCESSED, load_tables  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REVIEW_PERIOD_DAYS = CFG["review_policy"]["standard_cadence_days"]
Z_VALUES = {0.99: 2.326, 0.95: 1.645, 0.90: 1.282, 0.85: 1.036}


def service_level(criticality: str) -> float:
    return CFG["service_levels"].get(criticality, 0.90)


def build_replenishment_plan() -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = load_tables()
    sys.path.insert(0, str(ROOT / "db"))
    from inputs import load_forecasts  # noqa: E402
    fcst = load_forecasts()
    inv = tables["inventory_batches"]
    skus = tables["sku_master"]
    lanes = tables["lanes"]

    demand_stats = (
        fcst.groupby(["sku_id", "region"])
        .agg(
            mu_daily=("p50", "mean"),
            p10_mean=("p10", "mean"),
            p90_mean=("p90", "mean"),
        )
        .reset_index()
    )
    demand_stats["sigma_daily"] = (
        (demand_stats["p90_mean"] - demand_stats["p10_mean"]) / (2 * 1.2816)
    ).clip(lower=0)

    onhand = inv.groupby(["sku_id", "location"])["qty_units"].sum()
    lead = lanes[lanes["mode"] == "replenish"].set_index("to_location")[
        "lead_time_days"
    ]
    sku_idx = skus.set_index("sku_id")

    rows = []
    for _, r in demand_stats.iterrows():
        sku_id, region = r["sku_id"], r["region"]
        crit = sku_idx.loc[sku_id, "criticality"]
        unit_cost = int(sku_idx.loc[sku_id, "unit_cost_inr"])
        L = int(lead.get(region, 14))
        z = Z_VALUES[round(service_level(crit), 2)]
        mu, sigma = r["mu_daily"], r["sigma_daily"]

        ss = z * sigma * np.sqrt(L)
        target_position = mu * (L + REVIEW_PERIOD_DAYS) + ss
        oh = float(onhand.get((sku_id, region), 0))
        order_qty = max(0.0, target_position - oh)
        rows.append(
            {
                "sku_id": sku_id,
                "region": region,
                "criticality": crit,
                "lead_time_days": L,
                "service_level": service_level(crit),
                "mu_daily": round(mu, 1),
                "sigma_daily": round(sigma, 1),
                "safety_stock": int(round(ss)),
                "target_position": int(round(target_position)),
                "on_hand": int(oh),
                "order_qty": int(round(order_qty)),
                "order_value_inr": int(round(order_qty * unit_cost)),
                "days_of_supply_on_hand": round(oh / mu, 1) if mu > 0 else np.inf,
                "status": (
                    "stockout_risk"
                    if oh < mu * L
                    else ("low" if target_position - oh > ss else "ok")
                ),
            }
        )
    plan = pd.DataFrame(rows)
    return plan, inv


def main():
    plan, inv = build_replenishment_plan()

    # write plan to PostgreSQL OUTPUT table
    try:
        sys.path.insert(0, str(ROOT / "db"))
        import outputs as db_out

        db_out.write_replenishment(plan, db_out.current_as_of())
    except Exception as exc:
        print(f"[replenishment] DB write failed: {type(exc).__name__}: {exc}")

    print(plan["status"].value_counts())
    print("\ntop orders by value:")
    print(
        plan.nlargest(8, "order_value_inr")[
            ["sku_id", "region", "order_qty", "order_value_inr", "status"]
        ].to_string(index=False)
    )
    print("\ntotal replenishment value: ₹", f"{plan.order_value_inr.sum():,.0f}")


if __name__ == "__main__":
    main()
