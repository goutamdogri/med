# Frontend Spec — MedCare Supply Chain Control Tower (React)

**Audience:** frontend developer building the judge-facing SPA.
**Backend contract:** see `docs/backend-spec.md` — every widget below names its endpoint.
**Design goal:** this is not a dashboard dump. It tells the story of the problem
statement: *flu-season surge (+60%) starves Tier-2 cities while metro DCs sit on
expiring stock → our system senses it early, reallocates with expiry awareness, and
proves the ₹ impact.* Each page maps to one expected outcome.

---

## 0. App shell

- **Layout:** left sidebar (6 pages + pipeline health chip), top bar, content area.
  Dark theme works well for demo screens.
- **Top bar:** `asOf` date picker + "Data as of **2019-01-17** · run success · 2m28s ago"
  (from `GET /api/meta`). Changing `asOf` refetches everything on the current page —
  all endpoints accept it. Default = latest run.
- **Global state:** a tiny context holding `{ asOf, meta }`; pages fetch their own data
  (`react-query`/`swr` recommended for caching + loading states).
- **Charts:** Recharts (band charts via two stacked `<Area>` for P10–P90) or ECharts.
- **Numbers:** format INR as `₹12.1L` / `₹1.03Cr`, units with thousands separators.
- Severity palette: RED `#ef4444`, AMBER `#f59e0b`, OK/green `#22c55e`;
  policy colors consistent everywhere: proposed = blue, status_quo = gray.

### Pages ↔ problem-statement outcomes

| # | Page | Outcome it proves |
|---|---|---|
| 1 | Command Center | Overall value (availability ↑, wastage ↓) |
| 2 | Demand Sensing | Short-term forecast accuracy via leading indicators |
| 3 | Shortage Watchlist | Fewer stock-outs of critical SKUs |
| 4 | Expiry Rescue | Expiry-aware allocation, write-off reduction |
| 5 | Order Book | Replenishment plan output |
| 6 | Escalation Center | Review cadence / escalation process |

---

## 1. Command Center *(landing page)*

**Data:** `GET /api/kpi`, `GET /api/kpi/daily-curves`, `GET /api/kpi/writeoff-cumulative`,
`GET /api/inventory/aging`, `GET /api/runs?limit=1`.

| Widget | Spec |
|---|---|
| **KPI cards ×4** (top row) | Fill Rate %, Critical Fill Rate %, Write-off ₹, Stock-out site-days. Each card shows **proposed vs status_quo** with delta badge: `+9.1 pts`, `−54.6%`, `₹17.9L saved`. Delta math is backend-provided in `/api/kpi.improvement`. Green when proposed wins. |
| Served-vs-demand curve | Line chart, x = forecast date (42 days), y = units. Two shaded areas per policy or solid lines: `fulfilled` vs `demand`. Toggle: policy overlay vs side-by-side. |
| Write-off trajectory | Cumulative area chart of `expiredValueInr` per policy; vertical gap at final day annotated "₹ saved by expiry-aware planning". |
| Network utilization strip | Horizontal bars from aging summary: used vs capacity per location (`utilizationPct` if exposed; else compute from bucket totals). Metro DCs visibly over-stocked = the story hook. |
| Pipeline health chip (sidebar) | From `/api/runs`: status dot + duration. Proves live system. |

## 2. Demand Sensing

**Data:** `GET /api/demand/history`, `GET /api/forecasts`, `GET /api/flu`.

Controls row: SKU dropdown (32), region dropdown (6), **forecast-window slider 1–42 days**
(filters `horizonMax`) with label "Forecast window: N days".

| Widget | Spec |
|---|---|
| **Forecast band chart** (hero) | X = dates from `asOf+1`. Actuals line (`history`) up to asOf; then P10–P90 band (semi-transparent area between p10 and p90) with P50 line inside. Flu index overlaid as dashed secondary-axis line — visually shows flu curve *leading* demand spikes. This single chart answers "how do you sense demand?" |
| Sensing factor table | Per selected SKU's ATC code, avg across horizon: `momentumU` (show as `×1.34 momentum`), `fluRatio`, `senseAdjustment` (show as `+16.8% uplift`). Tooltip explaining each in one sentence. |
| Model mix donut | `lgbmWeight` vs `chronosWeight` from forecasts response — "ensemble of gradient boosting + foundation time-series model". |
| Accuracy note | Latest run WMAPE from `/api/runs` when non-null ("realized error on past forecasts"). |

## 3. Shortage Watchlist

**Data:** `GET /api/replenishment` (`status!=ok`, sorted DOS asc), `GET /api/replenishment/summary`, `GET /api/alerts?unackOnly=`.

| Widget | Spec |
|---|---|
| Status tiles | `stockout_risk` (red), `low` (amber), `ok` (gray) counts + total order value. Click tile filters grid. |
| **Risk grid** | Columns: SKU, brand (join client-side from meta), Region, Criticality chip, Days-of-Supply (color-coded <LT=red), Lead time, Safety stock, Order qty, Order value. Row click → drawer with mu/sigma/service level details + related alerts. |
| **Alert board** | RED/AMBER cards grouped by severity. Card: type icon, SKU@region, key facts (DOS, lead time, recommended units), action text, **Acknowledge button** → `PATCH /api/alerts/{id}/acknowledge` with logged-in user string (hardcode `csco@pharma.in` for demo). Acked cards collapse to a muted "✓ acknowledged" list — judges see a workflow, not just data. |

## 4. Expiry Rescue & Transfers

**Data:** `GET /api/transfers`, `GET /api/writeoffs`, `GET /api/inventory/aging`.

| Widget | Spec |
|---|---|
| Headline pair | "₹X moved out of expiry danger zone" (Σ transfer `valueSavedInr`) vs "₹Y residual risk" (Σ writeoff `residualValueInr`). Big numbers side by side. |
| Transfer table | Batch, SKU, From→To lane (arrow), Qty, Expires in N days (chip red <45), Reason badge (`expiry_rescue` teal / `shortage_rescue` purple), Value saved. Filter chips by reason. |
| Aging heatmap | Locations (rows) × buckets 0–30/31–60/61–90/90+ days (cols), cell color intensity = valueInr. Instantly shows metro DCs hoarding soon-to-expire stock pre-intervention. |
| Residual risk table | Top 10 batches by `residualValueInr`: what would still expire despite transfers (honest limitation = credibility). |

## 5. Order Book

**Data:** `GET /api/replenishment` (full, paginated), `GET /api/replenishment/summary`.

- Full grid with column sort/filters (status, criticality, region), CSV export button
  (client-side from current page set).
- Summary bar: total order value, orders placed vs no-op rows, split by criticality.
- Detail drawer: service level, safety-stock breakdown (mu, sigma, LT) so planners can audit the math.

## 6. Escalation Center

**Data:** `GET /api/digest`, `GET /api/alerts`.

| Widget | Spec |
|---|---|
| **Mode banner** | `DAILY_SURGE_MODE` (red banner, "review every day during flu surge") vs `WEEKLY_STANDARD` (calm blue, "weekly review"). Driven by `reviewMode`. |
| Surge stat chips | Red alert count, affected regions list (`surgeRegions`). |
| AI escalation brief | `digestText` rendered with whitespace-pre-wrap in a "Brief for CSCO" card, footer chip `Generated by gemma4:e2b local LLM`. Judges love on-device AI — keep attribution visible. |
| Cadence explainer (static) | Small card: surge mode = daily review until region leaves top-quartile ILI for 14 consecutive days; standard = weekly Monday review; any new RED alert escalates within 24 h regardless of cadence. |

---

## Cross-cutting UX rules

- **Loading:** skeleton blocks per widget (not full-page spinner). **Empty:** "No runs yet for this date — pick another date" with quick-jump to latest. **Error:** toast + retry; never blank screen.
- All fetches keyed by `[endpoint, asOf]` — switching date never shows stale mixed data.
- Keep raw JSON views behind a ⚙ toggle per widget (debug aid during integration, disable for demo).
- Mobile: not required; optimize for 1920×1080 projector.

## Judge-demo script (build pages in this order)

1. **Command Center** — open with the delta KPIs: "+9.1 pts fill rate, ₹17.9L waste avoided."
2. **Demand Sensing** — pick `N02BE-01 @ WH_INDORE`, slide window to 14 days, point at flu curve rising before demand does.
3. **Shortage Watchlist** — show red grid, click an alert, acknowledge it live.
4. **Expiry Rescue** — headline pair ₹saved vs ₹residual; heat map showing metro excess drained to Tier-2.
5. **Order Book** — one glance: "the plan your planners receive every morning."
6. **Escalation Center** — AI brief readout + cadence policy. Close the loop.

Fallback if API is down during judging: screenshots folder + the existing Streamlit app
(`./run_all.sh`) as plan B — mention it as the analyst workbench version.
