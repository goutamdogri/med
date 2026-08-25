from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def _load_tables_parquet() -> dict[str, pd.DataFrame]:
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


def load_tables() -> dict[str, pd.DataFrame]:
    """Primary source: MySQL pharma_sc INPUT/INPUT-ROLLING tables.
    Falls back to data/processed parquet when the DB is unreachable."""
    try:
        sys.path.insert(0, str(ROOT / "db"))
        from inputs import load_tables_from_db  # noqa: E402

        tables = load_tables_from_db()
        n = len(tables.get("demand_history", pd.DataFrame()))
        print(f"[features] inputs loaded from MySQL ({n} demand rows)")
        return tables
    except Exception as exc:  # pragma: no cover - operational fallback
        print(f"[features] MySQL unavailable ({type(exc).__name__}); falling back to parquet")
        return _load_tables_parquet()

# The model needs to know on any future forecast date "is there a promo running here?". This grid is the lookup table that answers that.
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

    # If 2 promotion runs, then the uplift occurs because of both promo, so keep only thr max uplift.
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


def build_covariates(tables: dict[str, pd.DataFrame], panel: pd.DataFrame) -> pd.DataFrame:
    """Region × day covariate grid covering demand history AND future dates with
    published flu readings (surveillance data arrives ahead of internal sales)."""
    dmax = panel["date"].max()
    flu = tables.get("flu_index")
    if flu is not None and len(flu):
        fmax = pd.to_datetime(flu["date"]).max()
        if pd.notna(fmax) and fmax > dmax:
            dmax = fmax
    dates = pd.date_range(panel["date"].min(), dmax, freq="D")
    regions = sorted(panel["region"].unique())
    grid = pd.MultiIndex.from_product(
        [dates, regions], names=["date", "region"]
    ).to_frame(index=False)
    if flu is not None and len(flu):
        f = pd.DataFrame({
            "date": pd.to_datetime(flu["date"]),
            "region": flu["region"],
            "flu_index": flu["flu_index"],
        }).drop_duplicates(["date", "region"])
        grid = grid.merge(f, on=["date", "region"], how="left")
    else:
        grid["flu_index"] = np.nan
    promo_grid = build_promo_grid(dates)
    grid = grid.merge(promo_grid, on=["date", "region"], how="left")
    grid["promo_uplift"] = grid["promo_uplift"].fillna(0.0)
    return grid


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


def melt_horizons(supervised: pd.DataFrame, panel: pd.DataFrame,
                  require_target: bool = True,
                  cov: pd.DataFrame | None = None) -> pd.DataFrame:
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
    if cov is None:
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
    subset = ["forecast_date", "futr_flu_index"]
    if require_target:
        subset.insert(0, "target")
    return out.dropna(subset=subset).reset_index(drop=True)
