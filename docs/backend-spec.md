# Backend Spec — Spring Boot API for MedCare Supply Chain Control Tower

**Audience:** backend developer building the REST API that serves the React frontend.
**You do NOT need to understand the ML pipeline.** Every endpoint below is a thin,
parameterized SQL query over the `pharma_sc` MySQL database. The Python ML pipeline
already computes everything and writes results into dedicated tables on a schedule.

---

## 1. System context

```
Kaggle/synthetic ingest → [ML pipeline: Python] ──writes──► pharma_sc (MySQL)
                                ▲                              │
                    daily rollover + monthly retrain           │ read-only
                                                               ▼
   React SPA ──HTTP──► Spring Boot API ◄───────────────────────┘
                          │
                          └── PATCH alerts acknowledge (only WRITE endpoint in v1)
```

- **Data flows one way**: pipeline → DB. The API is read-only except alert acknowledgment.
- All result tables carry an `as_of_date` column = the day the run happened. This lets
  the UI "time travel" and always shows a consistent snapshot.
- Fresh data arrives once per day via the rollover job; retraining happens monthly.
  Nothing here needs to call Python in v1 (see §7 for the future trigger proxy).

## 2. Database

| Item | Value |
|---|---|
| Host/port | `127.0.0.1:3306` |
| Database | `pharma_sc` |
| Driver | MySQL Connector/J (`com.mysql:mysql-connector-j`) |
| Charset | `utf8mb4` |

Connection pool: HikariCP, ~5 connections is plenty (read-only workload).
**Do not cache queries for more than ~5 minutes** — data refreshes daily, master data can cache longer (1 h).

### Table catalog (roles matter for joins)

| Role | Table | Grain | Used by |
|---|---|---|---|
| INPUT | `sku_master` | 1 row/SKU (32) | dropdowns, join for names/criticality/cost |
| INPUT | `locations` | 1 row/location (6) | dropdowns, map/filters |
| INPUT | `lanes` | supplier→DC, DC→WH lanes (11) | transfer context (lead times) |
| INPUT | `distributors`, `promo_calendar` | masters | context panels (optional v1) |
| ROLLING | `demand_history` | SKU×region×day (~353k rows) | actual-sales charts |
| ROLLING | `disease_burden_index` | region×day ILI score | flu overlay chart |
| ROLLING | `inventory_batches` | batch snapshot per `as_of_date` | aging buckets; **always filter to `MAX(as_of_date)`** |
| ROLLING | `warehouse_capacity_log` | location×weekly snapshot | utilization gauges/trend |
| DERIVED | `sku_market_share_monthly`, `location_demand_summary`, `sku_cost_history` | analytics | optional context cards |
| OUTPUT | `forecasts_final` | SKU×region×forecast_date×horizon (8064/run) | forecast band chart, sensing panel |
| OUTPUT | `replenishment_orders` | SKU×region per run (192) | order book, watchlist |
| OUTPUT | `transfer_plan`, `writeoff_risk` | per-batch recommendations | expiry-rescue page |
| OUTPUT | `simulation_daily` | SKU×region×day×policy (**16k rows/run**) | KPI + curves — AGGREGATE SERVER-SIDE |
| OUTPUT | `kpi_summary` | policy per run (2 rows) | headline KPI cards |
| OUTPUT | `alerts`, `alert_digest`, `rolling_run_log` | per run | escalation center, health |

Key dimension values you can hardcode as enums:
- `criticality`: `critical > high > standard > low`
- `replenishment_orders.status`: `ok | low | stockout_risk`
- `alerts.severity`: `RED | AMBER`; `type`: `shortage_risk | expiry_writeoff_risk`
- `policy`: `proposed | status_quo`
- regions: `DC_MUMBAI`, `DC_DELHI`, `WH_NAGPUR`, `WH_INDORE`, `WH_COIMBATORE`, `WH_LUCKNOW`

---

## 3. Conventions for EVERY endpoint

1. **`asOf` parameter** (ISO `yyyy-MM-dd`, optional). Default = latest run:
   ```sql
   SELECT MAX(as_of_date) FROM rolling_run_log WHERE status='success'
   ```
   Return the resolved value in every response as `"asOf"`.
2. Dates serialize as `yyyy-MM-dd`; decimals as numbers rounded to stored precision.
3. Pagination: `page` (0-based) + `size` (default 50, max 500) on all list endpoints →
   envelope `{ "content": [...], "page": 0, "size": 50, "totalElements": 192 }`.
4. Errors: standard `{ "status": 404, "error": "No run found for asOf=2019-05-01", ... }`.
5. CORS: allow the frontend origin on `/api/**`.
6. JSON field names: lowerCamelCase (map from snake_case columns).

---

## 4. Endpoints (v1)

### 4.1 `GET /api/meta`
One call the frontend makes on boot.

```json
{
  "asOf": "2019-01-17",
  "latestRun": {
    "ranAt": "2026-08-23T17:41:57",
    "status": "success",
    "modelsUsed": ["chronos", "lgbm"],
    "lgbmWeight": 0.503,
    "chronosWeight": 0.497,
    "wmape": null,
    "durationSeconds": 148
  },
  "skus":   [{ "skuId": "M01AB-01", "brandName": "Inflamac", "atcCode": "M01AB",
               "criticality": "standard", "unitCostInr": 20 }],   // all 32
  "regions": [{ "locationId": "DC_MUMBAI", "name": "Mumbai Metro DC", "type": "metro_dc" }]
}
```

### 4.2 `GET /api/kpi?asOf=`
Headline comparison. Source: `kpi_summary`.

```json
{
  "asOf": "2019-01-17",
  "proposed":    { "fillRatePct": 92.50, "criticalFillRatePct": 89.73, "stockoutUnits": 76038,
                   "criticalStockoutSitedays": 72, "writeoffValueInr": 1150512 },
  "statusQuo":   { "fillRatePct": 83.45, "criticalFillRatePct": 77.21, "stockoutUnits": 167674,
                   "criticalStockoutSitedays": 242, "writeoffValueInr": 2943641 },
  "improvement": { "fillRatePctDelta": 9.05, "criticalFillRatePctDelta": 12.52,
                   "stockoutUnitReductionPct": 54.6, "writeoffSavingInr": 1793129 }
}
```

### 4.3 `GET /api/kpi/daily-curves?asOf=`
Source: `simulation_daily` **aggregated server-side** — never return raw rows.

```sql
SELECT date, policy,
       SUM(demand) AS demand, SUM(fulfilled) AS fulfilled,
       SUM(unfulfilled) AS unfulfilled, SUM(expired_value_inr) AS expired_value_inr,
       AVG(ending_inventory) AS avg_ending_inventory
FROM simulation_daily
WHERE as_of_date = :asOf
GROUP BY date, policy ORDER BY date;
```

Response: `{ "asOf": "...", "series": { "proposed": [ {"date","demand","fulfilled","unfulfilled","expiredValueInr","avgEndingInventory"} ×42 ], "statusQuo": [ ... ] } }`

Also provide `GET /api/kpi/writeoff-cumulative?asOf=` → cumulative `SUM(expired_value_inr)` per policy per day (running total), used for the "₹ saved" area chart.

### 4.4 `GET /api/forecasts?asOf=&skuId=&region=&atcCode=&horizonMax=42&page=&size=`
Source: `forecasts_final`. Powers the forecast-band chart and sensing table.
`horizonMax` filters `horizon <= N` (the frontend's forecast-window slider — instant, no recompute).

```json
{ "content": [ {
    "skuId": "M01AB-01", "region": "DC_DELHI", "atcCode": "M01AB",
    "forecastDate": "2019-01-17", "horizon": 1,
    "p10": 45.61, "p50": 108.62, "p90": 180.63,
    "momentumU": 1.3362, "fluRatio": 43.8798, "senseAdjustment": 0.1681
  } ],
  "modelMix": { "lgbm": 0.503, "chronos": 0.497 }, "asOf": "2019-01-17", ...pagination }
```

### 4.5 `GET /api/demand/history?skuId=&region=&from=&to=`
Source: `demand_history`. Actuals only exist up to the current `asOf`.
→ `[ { "date": "2019-01-15", "units": 24506 } ]` (optionally grouped server-side if no sku/region filter: group by date, sum units — cap range at ~400 days).

### 4.6 `GET /api/flu?region=&from=&to=`
Source: `disease_burden_index` → `[ { "date": "2019-01-15", "region": "DC_MUMBAI", "indexValue": 2.34 } ]`.

### 4.7 `GET /api/replenishment?asOf=&status=&criticality=&region=&sort=dos&page=&size=`
Source: `replenishment_orders`. Sort key `dos` = `days_of_supply_on_hand ASC` (NULL/∞ last);
default sort `criticality ASC (critical first), dos ASC`.

```json
{ "content": [ {
    "skuId": "N02BE-01", "region": "WH_INDORE", "criticality": "critical",
    "leadTimeDays": 18, "serviceLevel": 0.99,
    "muDaily": 512.40, "sigmaDaily": 180.02, "safetyStock": 2178,
    "targetPosition": 24810, "onHand": 0, "orderQty": 21042,
    "orderValueInr": 147297, "daysOfSupplyOnHand": 0.0, "status": "stockout_risk"
  } ], ... }
```
Add `GET /api/replenishment/summary?asOf=` → counts + value by status & criticality:
`{ "byStatus": {"ok":144,"low":31,"stockout_risk":17}, "totalOrderValueInr": 11259023, "byCriticality": [...] }`

### 4.8 `GET /api/transfers?asOf=&reason=`
Source: `transfer_plan` (+ join `lanes` if you want carrier info).
`reason`: `expiry_rescue | shortage_rescue`.
Include summary: total units moved, total `valueSavedInr`, count by reason, by lane (`from→to`).
`srcDaysOfSupplyBefore` may be null (shortage rescues).

### 4.9 `GET /api/writeoffs?asOf=`
Source: `writeoff_risk`. Include totals: `SUM(residual_value_inr)` = residual exposure after optimization.
The frontend pairs this with transfers' `valueSavedInr` to show "saved vs residual".

### 4.10 `GET /api/inventory/aging?asOf=`
Source: `inventory_batches` filtered to `MAX(as_of_date)` ≤ requested. Bucket by days-to-expiry:

```sql
SELECT location, sku_id,
  CASE WHEN DATEDIFF(expiry_date, :asOf) <= 30 THEN 'd0_30'
       WHEN <= 60 THEN 'd31_60' WHEN <= 90 THEN 'd61_90' ELSE 'd90plus' END AS bucket,
  SUM(qty_units) AS units, SUM(qty_units*unit_cost_inr) AS valueInr
FROM inventory_batches WHERE as_of_date = (SELECT MAX(as_of_date) FROM inventory_batches)
GROUP BY location, sku_id, bucket;
```
Return both per-row and rolled-up-by-location shapes (frontend uses heatmap + stacked bars). Also expose `status` distribution (`healthy/watch/near_expiry_risk/stockout`).

### 4.11 `GET /api/alerts?asOf=&severity=&unackOnly=false&page=&size=`
Source: `alerts`. Parse `facts` JSON into a typed object (`Map<String,Object>` fine):

```json
{ "id": 1, "severity": "RED", "type": "shortage_risk",
  "skuId": "N02BE-01", "region": "WH_INDORE",
  "facts": { "criticality": "critical", "daysOfSupply": 0.0, "leadTimeDays": 18,
             "orderValueInr": 147297, "recommendedOrderUnits": 21042 },
  "action": "Expedite replenishment of 21042 units; consider transfer from metro DC.",
  "acknowledged": false, "acknowledgedBy": null, "acknowledgedAt": null,
  "createdAt": "..." }
```

### 4.12 `PATCH /api/alerts/{id}/acknowledge`
**Only write endpoint.**
Body: `{ "user": "csco@pharma.in" }` → sets `is_acknowledged=1, acknowledged_by=user, acknowledged_at=NOW()`.
Returns updated alert. Validate alert exists → else 404.

### 4.13 `GET /api/digest?asOf=`
Source: `alert_digest` (single row per run):

```json
{ "asOf": "2019-01-16", "reviewMode": "DAILY_SURGE_MODE",
  "surgeRegions": [],            // split comma string; empty array when null
  "redAlertCount": 17,
  "digestText": "Daily Escalation Brief for CSCO\n\n...",
  "modelUsed": "openai/gpt-oss-120b" }
```

### 4.14 `GET /api/runs?limit=10`
Source: `rolling_run_log` DESC by `as_of_date` — pipeline audit trail page.
Include `wmape` (may be null early on), `status`, `triggeredBy`, durations, model weights.

### 4.15 Masters (cached 1 h): `GET /api/master/{skus|locations|lanes}`
Straight dumps of those tables (snake_case → lowerCamelCase). Frontend uses for dropdowns/labels.

---

## 5. Suggested project layout

```
controller/  dto/  repo/  service/  config/
```
- Spring Data JPA entities are OK, but given read-mostly nature, **`JdbcTemplate` + DTO projection is simpler and faster to build** — recommend that over full JPA.
- One `@Repository` per domain (kpi, forecast, replenishment, allocation, alert).
- Global exception handler → JSON errors per §3.4.
- Actuator health at `/actuator/health` (CI deploy check hits this).

## 6. Performance notes
- Add composite indexes if slow: most needed ones already exist (see `db/schema.sql`); `simulation_daily` aggregation per run scans 16k rows — trivial.
- Enable HTTP caching headers on master endpoints only.
- Total payload for the heaviest dashboard view (curves + forecasts for one SKU) should stay < 200 KB thanks to server-side aggregation.

## 7. Pipeline triggers — stub now, wire later
v1: the daily/monthly jobs run independently (cron inside the ML container). Expose these
endpoints as **501-stubbed** so the contract exists:

```
POST /api/pipeline/rollover   { "asOf": "auto", "horizon": 42 }  → 202 { "runId": "..." }
POST /api/pipeline/retrain    {}                                  → 202
GET  /api/pipeline/status/{runId}
```
Later they will proxy to a small FastAPI sidecar next to the pipeline container
(`POST /run/daily`, `GET /status/{id}`) — same shapes, just forwarded. Do not shell out
to Python from Java.

## 8. Acceptance checklist
- [ ] `GET /api/meta` returns 32 SKUs + 6 regions + resolved asOf
- [ ] All list endpoints paginate + honor `asOf`
- [ ] `kpi/daily-curves` returns 42 points × 2 policies, aggregated (not 32k raw rows)
- [ ] Alert acknowledge persists + appears with `unackOnly=true` filtering
- [ ] Empty state: requesting a future `asOf` yields clean 404 JSON, not stack trace
