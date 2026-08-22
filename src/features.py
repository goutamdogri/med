from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def load_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    for name in [
        "demand_history",
        "sku_master",
        "locations",
        "lanes",
        "distributors",
        "promo_calendar",
        "flu_index",
        "inventory_batches",
    ]:
        path = PROCESSED / f"{name}.parquet"
        if path.exists():
            tables[name] = pd.read_parquet(path)
    return tables


def build_promo_grid(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    promos = pd.read_parquet(PROCESSED / "promo_calendar.parquet")
    all_regions = [loc["id"] for loc in CFG["locations"]]
    for _, p in promos.iterrows():
        regions = all_regions if p["regions"] == "ALL" else p["regions"].split(",")
        for region in regions:
            mask_dates = dates[(dates >= p["start_date"]) & (dates <= p["end_date"])]
            for d in mask_dates:
                rows.append((d, region, p["uplift"]))
    grid = pd.DataFrame(rows, columns=["date", "region", "promo_uplift"])
    return (
        grid.groupby(["date", "region"], as_index=False)["promo_uplift"]
        .max()
        if rows
        else pd.DataFrame(columns=["date", "region", "promo_uplift"])
    )


def build_panel(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    demand = tables["demand_history"]
    skus = tables["sku_master"][["sku_id", "criticality", "unit_cost_inr"]]
    locations = tables["locations"][["location_id", "type", "demand_share"]]
    flu = tables["flu_index"]

    panel = demand.merge(skus, on="sku_id", how="left").merge(
        locations, left_on="region", right_on="location_id", how="left"
    )
    panel = panel.merge(flu, on=["date", "region"], how="left")

    promo_grid = build_promo_grid(pd.DatetimeIndex(demand["date"].unique()))
    panel = panel.merge(promo_grid, on=["date", "region"], how="left")
    panel["promo_uplift"] = panel["promo_uplift"].fillna(0.0)

    panel = panel.sort_values(["sku_id", "region", "date"]).reset_index(drop=True)

    g = panel.groupby(["sku_id", "region"], sort=False)["units"]
    for lag in [1, 7, 14, 28]:
        panel[f"lag_{lag}"] = g.shift(lag)
    panel["roll_mean_7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
    panel["roll_std_7"] = g.transform(lambda s: s.shift(1).rolling(7).std())
    panel["roll_mean_28"] = g.transform(lambda s: s.shift(1).rolling(28).mean())
    panel["lag_364"] = g.shift(364)
    panel["lag_364"] = panel["lag_364"].fillna(panel["roll_mean_28"])

    dt = panel["date"]
    panel["dow"] = dt.dt.dayofweek
    panel["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    panel["month"] = dt.dt.month
    panel["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    crit_map = {"low": 0, "standard": 1, "high": 2, "critical": 3}
    panel["criticality_code"] = panel["criticality"].map(crit_map)
    panel["is_tier2"] = (panel["type"] == "tier2_wh").astype(int)

    return panel


HORIZONS = list(range(1, CFG['simulation']['horizon_days'] + 1))


def make_supervised(panel: pd.DataFrame, stride: int = 7) -> pd.DataFrame:
    work = panel.copy()
    for h in HORIZONS:
        work[f"target_{h}"] = work.groupby(["sku_id", "region"], sort=False)[
            "units"
        ].shift(-h)
    keep_cols = [
        "date",
        "sku_id",
        "region",
        "units",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "lag_364",
        "roll_mean_7",
        "roll_std_7",
        "roll_mean_28",
        "dow",
        "week_of_year",
        "month",
        "is_weekend",
        "promo_uplift",
        "flu_index",
        "criticality_code",
        "unit_cost_inr",
        "is_tier2",
        "demand_share",
        *[f"target_{h}" for h in HORIZONS],
    ]
    work = work[keep_cols]
    work = work.dropna(subset=["lag_28", "roll_mean_28"])
    if stride > 1:
        dates_sorted = np.sort(work["date"].unique())
        selected = set(dates_sorted[::stride])
        work = work[work["date"].isin(selected)]
    return work.reset_index(drop=True)


FEATURES = [
    "horizon",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "lag_364",
    "roll_mean_7",
    "roll_std_7",
    "roll_mean_28",
    "futr_dow",
    "futr_week_of_year",
    "futr_month",
    "futr_is_weekend",
    "futr_promo_uplift",
    "futr_flu_index",
    "criticality_code",
    "unit_cost_inr",
    "is_tier2",
    "demand_share",
]


def melt_horizons(supervised: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    static = [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "lag_364",
        "roll_mean_7",
        "roll_std_7",
        "roll_mean_28",
        "criticality_code",
        "unit_cost_inr",
        "is_tier2",
        "demand_share",
    ]
    cov = panel[["region", "date", "flu_index", "promo_uplift"]].drop_duplicates()
    for h in HORIZONS:
        piece = supervised[static].copy()
        piece.insert(0, "horizon", h)
        piece["target"] = supervised[f"target_{h}"].values
        piece["cutoff_date"] = supervised["date"].values
        piece["forecast_date"] = (
            pd.to_datetime(supervised["date"]) + pd.Timedelta(days=int(h))
        ).values
        piece["sku_id"] = supervised["sku_id"].values
        piece["region"] = supervised["region"].values

        fd = pd.DatetimeIndex(piece["forecast_date"])
        piece["futr_dow"] = fd.dayofweek
        piece["futr_week_of_year"] = fd.isocalendar().week.astype(int).values
        piece["futr_month"] = fd.month
        piece["futr_is_weekend"] = (fd.dayofweek >= 5).astype(int)

        piece = piece.merge(
            cov.rename(columns={"date": "forecast_date"}),
            on=["region", "forecast_date"],
            how="left",
        )
        piece = piece.rename(
            columns={"flu_index": "futr_flu_index", "promo_uplift": "futr_promo_uplift"}
        )
        pieces.append(piece)
    out = pd.concat(pieces, ignore_index=True)
    return out.dropna(subset=["target", "forecast_date", "futr_flu_index"]).reset_index(
        drop=True
    )
