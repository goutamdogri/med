# Replenishment Planning Engine — Deep Dive

**Source file:** `medcare_ml_model/src/replenishment.py`

This document explains the statistical inventory replenishment model: how it consumes probabilistic demand forecasts to calculate safety stock, target inventory positions, and order quantities for every SKU–region combination.

---

## Overview

The replenishment script is the **bridge between demand forecasting and supply chain execution**. It takes the probabilistic ensemble forecasts (P10/P50/P90) produced by `ensemble.py` and runs each SKU–region pair through a **Periodic Review (R, S) inventory policy** — the standard model used in pharmaceutical supply chains. The output is an actionable replenishment plan: how much of each SKU to order, from where, and at what cost.

The process follows this sequence:

```
Ensemble Forecasts (P10/P50/P90)
          +
Inventory Batches (on-hand stock)
          +
SKU Master (criticality, unit cost)
          +
Lanes (lead times by region)
          │
          ▼
  Demand Statistics (μ, σ per SKU-region)
          │
          ▼
  Safety Stock  =  Z × σ × √L
          │
          ▼
  Target Position  =  μ × (L + R)  +  Safety Stock
          │
          ▼
  Order Quantity  =  max(0, Target Position − On-Hand)
          │
          ▼
  PostgreSQL: replenishment_plan table
```

---

## Module-Level Constants

```python
REVIEW_PERIOD_DAYS = CFG["review_policy"]["standard_cadence_days"]
Z_VALUES = {0.99: 2.326, 0.95: 1.645, 0.90: 1.282, 0.85: 1.036}
```

| Constant | Description |
|---|---|
| `REVIEW_PERIOD_DAYS` (R) | How often the inventory is reviewed and orders are placed (e.g., every 7 or 14 days). Loaded from the project config YAML. This is the "R" in the (R, S) policy. |
| `Z_VALUES` | A lookup table mapping a **service level probability** to its corresponding **Z-score** (the standard normal quantile). Used to compute how large a safety stock buffer is needed to achieve that service level. |

### Z-Score Reference

| Service Level | Z-Score | Meaning |
|---|---|---|
| 99% | 2.326 | Stockouts expected only 1% of the time — used for critical/life-saving drugs |
| 95% | 1.645 | Common industry standard for essential medications |
| 90% | 1.282 | Default fallback for unconfigured SKUs |
| 85% | 1.036 | Acceptable for low-criticality, non-urgent items |

---

## Function Breakdown

### 1. `service_level(criticality)` — Criticality-to-SL Mapping

```python
def service_level(criticality: str) -> float:
    return CFG["service_levels"].get(criticality, 0.90)
```

Maps a SKU's `criticality` label (e.g., `"critical"`, `"essential"`, `"standard"`) to a numerical service level probability. The criticality→service-level mapping is defined in the project config YAML. If a criticality label is not found, it defaults to **90%** as a safe fallback.

This ensures that life-saving drugs automatically receive the highest inventory buffers without any per-SKU manual tuning.

---

### 2. `build_replenishment_plan()` — Core Calculation Engine

This is the main function that produces the replenishment plan. It follows four distinct phases:

#### Phase 1: Load Inputs

```python
tables = load_tables()
fcst   = load_forecasts()         # ensemble P10/P50/P90 from PostgreSQL
inv    = tables["inventory_batches"]
skus   = tables["sku_master"]
lanes  = tables["lanes"]
```

Four data sources are loaded:
- **`fcst`**: The latest ensemble forecasts — a table of `(sku_id, region, forecast_date, p10, p50, p90)` rows.
- **`inventory_batches`**: Current physical stock on hand, by SKU and location (may have multiple batches per SKU, e.g., expiry-date partitioned).
- **`sku_master`**: Product metadata including `criticality` and `unit_cost_inr`.
- **`lanes`**: Logistics lane definitions including `lead_time_days` per destination region (filtered to `mode == "replenish"`).

---

#### Phase 2: Derive Demand Statistics (μ and σ)

```python
demand_stats = (
    fcst.groupby(["sku_id", "region"])
    .agg(
        mu_daily   = ("p50", "mean"),
        p10_mean   = ("p10", "mean"),
        p90_mean   = ("p90", "mean"),
    )
    .reset_index()
)
demand_stats["sigma_daily"] = (
    (demand_stats["p90_mean"] - demand_stats["p10_mean"]) / (2 * 1.2816)
).clip(lower=0)
```

This step summarises the 28-day horizon forecast down to two parameters per SKU–region:

**`mu_daily` (μ) — Expected Daily Demand:**
- The mean of the P50 (median) forecast across all 28 forecast days.
- This is the best-guess average daily demand and is used for the "cycle stock" calculation (the stock consumed during lead time and the review period).

**`sigma_daily` (σ) — Daily Demand Uncertainty:**
- Derived from the **interquartile range of the forecast distribution**: `(P90 − P10) / (2 × 1.2816)`.
- The value `1.2816` is the Z-score at the 90th percentile. Since P90 − P10 spans from the 10th to 90th percentile (a range of `2 × 1.2816` standard deviations), dividing by `2 × 1.2816` back-calculates the implied standard deviation (σ) of the forecast distribution.
- This is a clean, model-agnostic way to extract uncertainty: the ensemble's spread *is* the uncertainty estimate — no separate variance model is needed.
- `.clip(lower=0)` ensures σ is never negative.

> **Why this matters:** Using the model's own P10/P90 spread to compute σ means the safety stock is directly informed by the ensemble's confidence. A product with high forecast uncertainty (wide P10–P90 gap) automatically gets a larger safety buffer.

---

#### Phase 3: The Core Inventory Formula (per SKU–region loop)

```python
for _, r in demand_stats.iterrows():
    sku_id, region = r["sku_id"], r["region"]
    crit      = sku_idx.loc[sku_id, "criticality"]
    unit_cost = int(sku_idx.loc[sku_id, "unit_cost_inr"])
    L         = int(lead.get(region, 14))
    z         = Z_VALUES[round(service_level(crit), 2)]
    mu, sigma = r["mu_daily"], r["sigma_daily"]

    ss              = z * sigma * np.sqrt(L)
    target_position = mu * (L + REVIEW_PERIOD_DAYS) + ss
    oh              = float(onhand.get((sku_id, region), 0))
    order_qty       = max(0.0, target_position - oh)
```

For each SKU–region pair, the script resolves:
- `L` (lead time in days): Looked up from the logistics `lanes` table for that region. Defaults to **14 days** if no lane is defined.
- `z` (Z-score): Resolved from `Z_VALUES` using the SKU's service level.
- `mu`, `sigma`: From Phase 2.

Then it applies the three-step inventory formula:

---

##### Formula Step 1: Safety Stock (SS)

```
SS = Z × σ × √L
```

| Symbol | Value | Description |
|---|---|---|
| Z | e.g., 1.645 | Z-score corresponding to the desired service level |
| σ | `sigma_daily` | Daily demand standard deviation (from forecast uncertainty) |
| √L | e.g., √14 ≈ 3.74 | Square root of lead time; demand uncertainty accumulates as √time |

**Interpretation:** Safety stock is the buffer that absorbs demand variability *during the lead time*. The `√L` scaling comes directly from statistical theory — if daily demand is i.i.d. with std σ, the std of total demand over L days is `σ × √L`. The Z-score then sets how many standard deviations of buffer to hold.

**Example:**
- σ = 50 units/day, L = 14 days, service level = 95% (Z = 1.645)
- SS = 1.645 × 50 × √14 ≈ **308 units**

---

##### Formula Step 2: Target Inventory Position (S)

```
S = μ × (L + R) + SS
```

| Symbol | Value | Description |
|---|---|---|
| μ | `mu_daily` | Expected daily demand |
| L | `lead_time_days` | Days for the order to arrive |
| R | `REVIEW_PERIOD_DAYS` | Days until the next review/order opportunity |
| SS | From Step 1 | Safety stock buffer |

**Interpretation:** The target position `S` is the inventory level you want to be at *right now*, so that after waiting L days for the order and then R days until the next review, you still have enough stock (plus a buffer). It's the "order-up-to" level of the classic (R, S) periodic review policy.

- `μ × L` = stock consumed while the order is in transit.
- `μ × R` = stock consumed during the next review cycle (until you can order again).
- `+ SS` = the safety buffer for unexpected demand spikes.

**Example (continuing):**
- μ = 100 units/day, L = 14 days, R = 7 days, SS = 308 units
- S = 100 × (14 + 7) + 308 = 2,100 + 308 = **2,408 units**

---

##### Formula Step 3: Order Quantity

```
Order Qty = max(0, S − On-Hand)
```

- If current on-hand stock already meets or exceeds the target position, the order quantity is **zero** (no order needed).
- Otherwise, order exactly enough to top up to the target position.

On-hand stock is the sum of all inventory batch quantities for that SKU at that region/location.

---

#### Phase 4: Derived Diagnostics and Status Flags

Each row in the output plan also includes:

| Column | Formula / Logic | Business Meaning |
|---|---|---|
| `order_value_inr` | `order_qty × unit_cost_inr` | Total value of the proposed purchase order in Indian Rupees |
| `days_of_supply_on_hand` | `on_hand / mu_daily` (∞ if μ=0) | How many days the current stock will last at average demand |
| `status` | See below | A traffic-light indicator for inventory health |

**Status Logic:**

```python
"stockout_risk"  if oh < mu * L               # Stock won't survive lead time
"low"            if target_position - oh > ss  # Order backlog is large
"ok"             otherwise
```

| Status | Condition | Action |
|---|---|---|
| `stockout_risk` | On-hand < μ × L — stock will be depleted before the order even arrives | **Urgent**: potential stockout during transit |
| `low` | The gap to fill (S − OH) exceeds safety stock — we are drawing down our buffer | **Warning**: order needed soon |
| `ok` | Comfortable position; safety buffer is intact | No immediate action required |

---

## `main()` — Execution and Output

```python
def main():
    plan, inv = build_replenishment_plan()
    db_out.write_replenishment(plan, db_out.current_as_of())

    print(plan["status"].value_counts())
    print(plan.nlargest(8, "order_value_inr")[...])
    print("total replenishment value: ₹", f"{plan.order_value_inr.sum():,.0f}")
```

1. **Builds the plan** and writes it to the `replenishment_plan` PostgreSQL table via `write_replenishment()`.
2. **Prints a status summary** — a quick count of how many SKU–regions are `ok`, `low`, or `stockout_risk`.
3. **Prints the top 8 orders by value** — useful for immediate prioritisation in procurement.
4. **Prints the total replenishment value in INR** — a headline figure for budget planning.

If the DB write fails (e.g., connection error), it logs the error but does not crash, allowing the plan to still be printed to the console.

---

## Output DataFrame Schema

| Column | Type | Description |
|---|---|---|
| `sku_id` | str | Product identifier |
| `region` | str | Geographic region / location |
| `criticality` | str | SKU criticality tier (e.g., `critical`, `essential`, `standard`) |
| `lead_time_days` | int | Days from order placement to delivery for this region |
| `service_level` | float | Target service level (e.g., 0.95) |
| `mu_daily` | float | Expected daily demand (units) |
| `sigma_daily` | float | Daily demand standard deviation (units) |
| `safety_stock` | int | Calculated buffer stock (units) |
| `target_position` | int | Order-up-to inventory level (units) |
| `on_hand` | int | Current physical stock (units) |
| `order_qty` | int | Recommended order quantity (units) |
| `order_value_inr` | int | Total value of the order (₹) |
| `days_of_supply_on_hand` | float | Current stock duration at average demand |
| `status` | str | `ok` / `low` / `stockout_risk` |

---

## Key Design Decisions

| Design Choice | Rationale |
|---|---|
| **σ derived from P10/P90 spread** | Avoids needing a separate residual-based variance model. The ensemble's own uncertainty directly drives safety stock, making the system self-consistent. |
| **`√L` scaling for safety stock** | Statistically correct under i.i.d. daily demand assumption. Total demand variance over L days = L × σ², so std = σ√L. |
| **Criticality-tiered service levels** | Pharmaceutical supply chains must prioritise life-critical drugs. Hard-coding equal service levels across all products would be clinically and commercially irresponsible. |
| **Default lead time of 14 days** | A conservative fallback when no lane is defined for a region, preventing the system from generating under-stocked plans for unknown lanes. |
| **`max(0, ...)` on order_qty** | Prevents negative orders; a product that is already over-stocked should never trigger a "return" recommendation from this module. |
| **Periodic Review (R, S) policy** | Simpler to execute than continuous review (s, Q) in pharmaceutical distribution, where orders are typically batched on a fixed schedule rather than triggered individually per stockout event. |
