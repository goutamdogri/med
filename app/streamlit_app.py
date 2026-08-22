from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import CFG, PROCESSED  # noqa: E402

st.set_page_config(page_title="MedCare Demand Control Tower", page_icon="💊", layout="wide")


@st.cache_data
def load(name) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


@st.cache_data
def load_csv(name) -> pd.DataFrame:
    return pd.read_csv(PROCESSED / f"{name}.csv")


@st.cache_data
def load_meta() -> dict:
    return yaml.safe_load((PROCESSED / "meta.json").read_text())


@st.cache_data
def load_json(name) -> dict:
    return json.loads((PROCESSED / f"{name}.json").read_text())


meta = load_meta()
demand = load("demand_history")
skus = load("sku_master")
locations = load("locations")
inventory = load("inventory_batches")
fcst = load("forecasts_final")
sim = load("simulation_daily")
kpi = load_csv("kpi_summary")
repl = load("replenishment_orders")
transfers = load("transfer_plan")
writeoffs = load("writeoff_risk")
alerts_payload = load_json("alerts")

page = st.sidebar.radio(
    "Navigate",
    [
        "📌 Executive Summary",
        "📡 Demand Sensing",
        "🔄 Allocation & Transfers",
        "📦 Replenishment Plan",
        "🚨 Escalation Center",
        "📥 Data & Retrain",
    ],
)
st.sidebar.caption(f"Network as-of **{meta['as_of_date']}** · {meta['n_skus']} SKUs × {meta['n_locations']} locations · INR")

if page == "📌 Executive Summary":
    st.title("MedCare Pharma — Demand Sensing & Replenishment Control Tower")
    prop = kpi[kpi["policy"] == "proposed"].iloc[0]
    sq = kpi[kpi["policy"] == "status_quo"].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fill Rate", f"{prop.fill_rate_pct:.1f}%", f"{prop.fill_rate_pct - sq.fill_rate_pct:+.1f} pts vs status quo")
    c2.metric("Critical SKU Fill", f"{prop.critical_fill_rate_pct:.1f}%", f"{prop.critical_fill_rate_pct - sq.critical_fill_rate_pct:+.1f} pts")
    c3.metric("Write-offs (42d)", f"₹{prop.writeoff_value_inr/1e5:.1f}L", f"−₹{(sq.writeoff_value_inr - prop.writeoff_value_inr)/1e5:.1f}L saved")
    c4.metric("Critical Stockout Site-Days", f"{int(prop.critical_stockout_sitedays)}", f"{int(prop.critical_stockout_sitedays - sq.critical_stockout_sitedays):+d}")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure()
        for pol, label in [("status_quo", "Status Quo"), ("proposed", "Proposed")]:
            d = sim[sim["policy"] == pol].groupby("date")[["demand", "fulfilled"]].sum().reset_index()
            fig.add_trace(go.Scatter(x=d["date"], y=d["fulfilled"], name=f"{label} served", stackgroup=None if pol == "status_quo" else None))
            fig.add_trace(go.Scatter(x=d["date"], y=d["demand"], name=f"{label} demand", line=dict(dash="dot"), opacity=0.5))
        fig.update_layout(title="Daily demand served vs demand", height=380, yaxis_title="units")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = go.Figure()
        for pol in ["status_quo", "proposed"]:
            d = sim[sim["policy"] == pol]
            daily_wo = d.groupby("date")["expired_value_inr"].first().cumsum()
            fig.add_trace(go.Scatter(x=daily_wo.index, y=daily_wo.values, name=pol, mode="lines"))
        fig.update_layout(title="Cumulative expiry write-offs (₹)", height=380, yaxis_title="₹")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast model leaderboard (same-origin backtest)")
    torch_scores = load_csv("backtest_torch").rename(columns={"model": "model", "wmape_asof_origin": "wmape"})
    lgb_bt = load_csv("backtest_lgbm")
    lgb_row = lgb_bt[(lgb_bt["model"] == "lightgbm") & (lgb_bt["origin"] == meta["as_of_date"])][["model", "wmape"]]
    board = pd.concat([torch_scores[["model", "wmape"]], lgb_row], ignore_index=True).sort_values("wmape")
    ens = pd.read_parquet(PROCESSED / "forecasts_final.parquet")
    fig = px.bar(board, x="model", y="wmape", color="wmape", text_auto=".3f",
                 color_continuous_scale="RdYlGn_r", labels={"wmape": "WMAPE (lower is better)"})
    fig.update_layout(height=320, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

elif page == "📡 Demand Sensing":
    st.title("Demand Sensing Explorer")
    sku_ids = sorted(fcst["sku_id"].unique())
    regions = sorted(fcst["region"].unique())
    c1, c2 = st.columns(2)
    sel_sku = c1.selectbox("SKU", sku_ids, index=sku_ids.index("R03-03"))
    sel_region = c2.selectbox("Location", regions)

    hist = demand[(demand["sku_id"] == sel_sku) & (demand["region"] == sel_region)]
    f = fcst[(fcst["sku_id"] == sel_sku) & (fcst["region"] == sel_region)].sort_values("horizon")

    fig = go.Figure()

    as_of = pd.Timestamp(meta["as_of_date"])
    hist_tail = hist[hist["date"] <= as_of].tail(120)

    fig.add_trace(go.Scatter(x=hist_tail["date"], y=hist_tail["units"], name="Actual history", line=dict(color="grey")))
    fig.add_vline(x=as_of, line_dash="dash", annotation_text="as-of")
    fig.add_trace(go.Scatter(x=f["forecast_date"], y=f["p50"], name="Ensemble p50 (sensed)", line=dict(color="crimson", width=3)))
    fig.add_trace(go.Scatter(x=list(f["forecast_date"]) + list(f["forecast_date"][::-1]),
                             y=list(f["p90"]) + list(f["p10"][::-1]), fill="toself",
                             fillcolor="rgba(220,20,60,0.15)", line=dict(width=0), name="80% interval"))
    for m, colr, label in [
        ("tft_p50", "orange", "tft"),
        ("chronos_p50", "green", "chronos"),
        ("nhits_p50", "purple", "nhits"),
        ("lgbm_p50", "blue", "lgbm"),
        ("p50_lgbm", "blue", "lgbm"),
    ]:
        if m in f.columns:
            fig.add_trace(go.Scatter(x=f["forecast_date"], y=f[m], name=label, line=dict(color=colr, width=1, dash="dot"), opacity=0.7))
    fig.update_layout(height=430, yaxis_title="units/day", title=f"{sel_sku} @ {sel_region}")
    st.plotly_chart(fig, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sensed uplift (wk-1)", f"{f['sense_adjustment'].iloc[0]*100:+.1f}%", help="Damped sensing adjustment applied to raw ensemble")
    m2.metric("Momentum u", f"{f['momentum_u'].iloc[0]:.2f}", help="Last-7d vs prior-28d demand ratio")
    m3.metric("Flu index ratio", f"{f['flu_ratio'].iloc[0]:.2f}", help="Flu-index today vs 14d ago")
    m4.metric("Avg daily fcst (42d)", f"{f['p50'].mean():.0f}")

    st.info("Leading indicators baked into every forecast: **flu surveillance index** (leads demand by ~8 days), promo calendar, recent velocity momentum.")

    st.subheader("What drives the LightGBM baseline?")
    imp = load_csv("lgbm_feature_importance").head(10)
    fig = px.bar(imp[::-1], x="gain", y="feature", orientation="h", title="Feature importance (gain)")
    fig.update_layout(height=340)
    st.plotly_chart(fig, use_container_width=True)

elif page == "🔄 Allocation & Transfers":
    st.title("Expiry-Aware Allocation & Transfer Planner")

    c1, c2, c3 = st.columns(3)
    rescue = transfers[transfers["reason"] == "expiry_rescue"]
    shortage = transfers[transfers["reason"] == "shortage_rescue"]
    c1.metric("Expiry-rescue transfers", len(rescue), f"₹{rescue['value_saved_inr'].sum()/1e5:.1f}L rescued")
    c2.metric("Shortage-rescue transfers", len(shortage), f"{int(shortage['qty_units'].sum()):,} units to Tier-2")
    c3.metric("Residual write-off exposure", f"₹{writeoffs['residual_value_inr'].sum()/1e5:.1f}L", "after all actions")

    st.subheader("Recommended transfers")
    view = transfers.copy()
    view["expiry_date"] = pd.to_datetime(view["expiry_date"]).dt.date
    st.dataframe(
        view[["sku_id", "from_location", "to_location", "qty_units", "reason", "days_to_expiry", "transfer_lead_days", "value_saved_inr"]],
        use_container_width=True, height=360,
    )

    col1, col2 = st.columns(2)
    with col1:
        agg = transfers.groupby(["from_location", "to_location"])["qty_units"].sum().reset_index()
        fig = px.bar(agg, x="to_location", y="qty_units", color="from_location", barmode="group",
                     title="Transfer volumes by lane")
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        wo_top = writeoffs.nlargest(10, "residual_value_inr")
        fig = px.bar(wo_top, x="residual_value_inr", y="batch_id", orientation="h",
                     color="location", title="Top residual write-off risks (no viable transfer)",
                     labels={"residual_value_inr": "₹ at risk"})
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("FEFO compliance view — batches by expiry bucket")
    inv = inventory.copy()
    as_of = pd.Timestamp(meta["as_of_date"])
    inv["days_to_expiry"] = (inv["expiry_date"] - as_of).dt.days
    bins = [-1, 30, 60, 90, 180, 10000]
    labels = ["<30d", "30-60d", "60-90d", "90-180d", ">180d"]
    inv["bucket"] = pd.cut(inv["days_to_expiry"], bins=bins, labels=labels)
    piv = inv.pivot_table(index="location", columns="bucket", values="qty_units", aggfunc="sum", observed=True).fillna(0)
    fig = px.imshow(piv.values, x=list(piv.columns), y=list(piv.index), aspect="auto",
                    color_continuous_scale="OrRd", labels=dict(color="units"),
                    title="Inventory units by remaining shelf life")
    fig.update_layout(height=340)
    st.plotly_chart(fig, use_container_width=True)

elif page == "📦 Replenishment Plan":
    st.title("Replenishment Orders — Next Cycle")
    total_val = repl["order_value_inr"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total order value", f"₹{total_val/1e6:.2f}M")
    c2.metric("Positions to order", int((repl["order_qty"] > 0).sum()))
    c3.metric("Stockout-risk positions", int((repl["status"] == "stockout_risk").sum()))

    f1, f2 = st.columns(2)
    sel_region = f1.multiselect("Location", sorted(repl["region"].unique()), default=sorted(repl["region"].unique()))
    sel_status = f2.multiselect("Status", ["stockout_risk", "low", "ok"], default=["stockout_risk", "low"])
    view = repl[repl["region"].isin(sel_region) & repl["status"].isin(sel_status)]
    st.dataframe(
        view.sort_values(["criticality", "order_value_inr"], ascending=[True, False])[
            ["sku_id", "region", "criticality", "mu_daily", "on_hand", "days_of_supply_on_hand",
             "lead_time_days", "safety_stock", "target_position", "order_qty", "order_value_inr", "status"]
        ],
        use_container_width=True, height=420,
    )
    st.caption("Order-up-to logic: target = μ×(L + 7-day review) + z×σ×√L, σ from ensemble P90–P10 band; service levels 99%/95%/90% by criticality.")

elif page == "📥 Data & Retrain":
    import tempfile

    from ingest import load_any_csv

    st.title("Bring Your Own Data")
    st.caption(
        "Upload daily sales as **wide** CSV (date + one column per ATC category, Kaggle format) "
        "or **long** CSV (date, atc_code, units). The full pipeline — features, models, plans, "
        "simulation, alerts — reruns on your data."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("History rows", f"{len(demand):,}")
    c2.metric("Date range", f"{demand['date'].min():%Y-%m-%d} → {demand['date'].max():%Y-%m-%d}")
    log_p = PROCESSED / "retrain_log.json"
    if log_p.exists():
        last = json.loads(log_p.read_text())
        c3.metric("Last retrain", last.get("finished", last["started"]), "OK" if last["ok"] else "FAILED")

    up = st.file_uploader("Sales CSV", type=["csv"])
    skip_torch = st.checkbox(
        "Skip neural retrain (fast mode)",
        value=True,
        help="Reuses N-HiTS/TFT/Chronos forecasts only if they match this dataset; otherwise falls back to LightGBM-only ensemble.",
    )

    if up is not None:
        tmp = Path(tempfile.mkdtemp()) / up.name
        tmp.write_bytes(up.getvalue())
        try:
            data, report = load_any_csv(tmp)
        except Exception as exc:
            st.error(f"Could not parse file: {exc}")
            data = None

        if data is not None:
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Format", report["format"])
            a2.metric("Days", report["rows"])
            a3.metric("Categories", len(report["categories"]))
            a4.metric("Coverage", f"{report['date_min']} → {report['date_max']}")

            if report["warnings"]:
                st.warning("\n".join(f"• {w}" for w in report["warnings"]))
            else:
                st.success("Validation clean.")
            st.dataframe(data.head(10), use_container_width=True)

            stale = str(pd.Timestamp(report["date_max"]).date()) < str(demand["date"].max().date())
            if stale:
                st.info("Uploaded history ends before the current dataset — you can still install it.")

            force = False
            if not report["ok"]:
                force = st.checkbox("Install anyway (degraded accuracy expected)")
            go = st.button("Install & retrain full pipeline", type="primary", disabled=not (report["ok"] or force))

            if go:
                from ingest import install

                install(tmp)
                status = st.status("Retraining…", expanded=True)
                out = status.empty()
                proc = subprocess.Popen(
                    [sys.executable, str(ROOT / "src" / "retrain.py")] + (["--skip-torch"] if skip_torch else []),
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                lines: list[str] = []
                for line in proc.stdout:
                    lines.append(line.rstrip())
                    out.code("\n".join(lines[-25:]))
                proc.wait()
                if proc.returncode == 0:
                    status.update(label="Retrain complete ✔", state="complete", expanded=False)
                    st.cache_data.clear()
                    st.success("All artifacts rebuilt. Pages now reflect your data.")
                    if st.button("Refresh"):
                        st.rerun()
                else:
                    status.update(label="Retrain failed — previous artifacts restored", state="error")
                    st.error("See log above; processed outputs were rolled back to the last good state.")

elif page == "🚨 Escalation Center":
    st.title("Shortage Escalation & Review Cadence")
    cadence = alerts_payload["review_cadence"]
    alerts = alerts_payload["alerts"]

    mode_color = {"DAILY_SURGE_MODE": "🔴", "WEEKLY_STANDARD": "🟢"}[cadence["mode"]]
    st.markdown(
        f"""### {mode_color} Review mode: `{cadence['mode']}`
        Cadence: **every {cadence['surge_cadence_days'] if 'SURGE' in cadence['mode'] else cadence['standard_cadence_days']} day(s)** · Red alerts: **{cadence['red_alerts']}** · Surge regions: *{', '.join(cadence['surge_regions']) or 'none'}*"""
    )
    st.caption("Trigger: sensed uplift > 20% or ≥5 red alerts ⇒ escalate from weekly S&OP to daily surge reviews.")

    sev_filter = st.multiselect("Severity", ["RED", "AMBER"], default=["RED", "AMBER"])
    type_filter = st.multiselect(
        "Type", sorted({a["type"] for a in alerts}), default=sorted({a["type"] for a in alerts})
    )
    rows = []
    for a in alerts:
        if a["severity"] in sev_filter and a["type"] in type_filter:
            rows.append({
                "severity": a["severity"],
                "type": a["type"],
                "sku": a["sku_id"],
                "region": a["region"],
                "key_fact": ", ".join(f"{k}={v}" for k, v in list(a["facts"].items())[:2]),
                "recommended_action": a["action"],
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=380)

    st.divider()
    st.subheader("🤖 AI Escalation Brief (local Gemma 4B)")
    digest_path = PROCESSED / "alert_digest.md"
    st.markdown(digest_path.read_text())
    if st.button("Regenerate brief"):
        with st.spinner("Gemma is writing..."):
            from alerts import gemma_digest

            kpi_d = kpi.set_index("policy").to_dict("index")
            txt = gemma_digest(alerts, kpi_d, cadence)
            digest_path.write_text(txt)
            st.rerun()

    st.divider()
    st.subheader("Escalation policy")
    esc = CFG["review_policy"]["escalation"]
    st.markdown(
        f"""
| Trigger | Action |
|---|---|
| Days of supply < **{esc['red_stock_days']}** | 🔴 RED — expedite order + emergency transfer |
| Days of supply < **{esc['amber_stock_days']}** | 🟠 AMBER — pull forward next PO |
| Sensed uplift > **{CFG['review_policy']['surge_trigger_sensed_uplift']:.0%}** | Switch region to daily review |
| Write-off exposure > ₹{esc['transfer_approval_value_inr']:,}/batch | CSCO approval for markdown/donation |
"""
    )
