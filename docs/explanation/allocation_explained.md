# Inventory Allocation Engine — Deep Dive

**Source file:** `medcare_ml_model/src/allocation.py`

This document explains the inventory allocation and expiry-rescue system: how it identifies at-risk stock, routes it optimally to locations where it will be consumed before it expires, and provides a secondary mechanism to rescue shortages via DC-to-region transfers.

---

## Overview

The allocation script solves **two distinct problems simultaneously**:

1. **Expiry Rescue (Proactive):** Identify inventory batches that will expire before they can be sold at their current location, and transfer them to locations with higher demand — saving value that would otherwise be written off.
2. **Shortage Rescue (Reactive):** Detect locations already flagged as `stockout_risk` or `low` in the replenishment plan, and initiate emergency transfers from Distribution Centres (DCs) to bridge the gap until the regular replenishment order arrives.

The full execution flow is:

```
Ensemble Forecasts (P50 per SKU-region)
        +
Inventory Batches (qty, expiry_date, unit_cost)
        +
Replenishment Plan (status, target_position, on_hand)
        +
Lanes (transfer routes with lead times)
        │
        ▼
Step 1: Score each batch for expiry risk
        │   leftover = qty - (mu × days_to_expiry)
        ▼
Step 2: EXPIRY RESCUE — find receiving destinations for at-risk batches
        │   Sort candidates by value saved (usable × unit_cost)
        ▼
Step 3: SHORTAGE RESCUE — DC→region emergency transfers for low/stockout SKUs
        │
        ▼
Step 4: Compute residual write-off exposure after all transfers
        │
        ▼
PostgreSQL: transfer_plan, writeoff_risk tables
```

---

## Module-Level Setup

```python
from features import CFG, PROCESSED, load_tables
ROOT = Path(__file__).resolve().parents[1]
```

No module-level numeric constants — all policy thresholds are embedded inline in the functions and explained below.

---

## Function Breakdown

### 1. `shortage_rescue_transfers(inv, repl, mu)` — Emergency DC Transfers

This function handles the **reactive** shortage rescue: when a location is already running low, it looks for a nearby Distribution Centre (DC) that has surplus stock and initiates an expedited transfer.

#### Step 1: Identify Needy Locations

```python
need_rows = repl[repl["status"].isin(["stockout_risk", "low"])]
```

Filters the replenishment plan to only those SKU–region pairs flagged as problematic. These are the destinations that need emergency stock.

#### Step 2: Build Transfer Lane Index

```python
transfer_lanes = transfer_lanes[transfer_lanes["mode"] == "transfer"]
lanes_by_dest = {
    dest: (src, lead)
    for src, dest, lead in zip(...)
    if src.startswith("DC_")
}
```

From the lanes table, filters to `mode == "transfer"` (lateral DC→region moves, as opposed to `mode == "replenish"` which are supplier→DC orders). Builds a fast `destination → (source DC, lead_time)` lookup, but **only from DC sources** (locations starting with `"DC_"`). This prevents store-to-store transfers which could create cascading shortages.

#### Step 3: Evaluate Each Shortage

For each needy SKU–region, the function evaluates whether the source DC can afford to give stock:

```python
src_dos       = oh_src / mu_src                        # DC's days of supply
min_keep_days = max(15.0, r["lead_time_days"])         # DC must keep at least this much
giveable_units = oh_src - mu_src * min_keep_days       # surplus above the DC's own safety buffer
dest_gap      = max(0.0, target_position - on_hand)    # how much the destination needs
qty           = min(giveable_units, dest_gap, mu_dst * 60)  # conservative cap at 60 days of dest demand
```

| Variable | Formula | Purpose |
|---|---|---|
| `src_dos` | `oh_src / mu_src` | Days of supply the source DC currently holds |
| `min_keep_days` | `max(15, lead_time_days)` | The minimum stock the DC must keep for itself — at least 15 days, or the destination's lead time if longer |
| `giveable_units` | `oh_src − mu_src × min_keep_days` | Stock the DC can release without endangering itself |
| `dest_gap` | `target_position − on_hand` | How short the destination is |
| `qty` | `min(giveable, gap, mu_dst × 60)` | Capped to the smallest of: what the DC can give, what the destination needs, or 60 days of destination demand |

**Gate conditions — transfer is skipped if:**
- `qty < 10` units — transfers below this are operationally impractical.
- `src_dos < min_keep_days + 5` — the DC doesn't have enough headroom (needs at least 5 extra days of buffer above its own minimum).
- `mu_src <= 0 or mu_dst <= 0` — if demand data is missing for either side, the transfer cannot be safely sized.

**Output fields include:**
- `reason: "shortage_rescue"` — tags these transfers distinctly from expiry rescues.
- `src_days_of_supply_before` — captured for auditing and what-if analysis.
- `batch_id: None` — shortage rescues are not tied to a specific batch (any available stock can be used).

---

### 2. `build_allocation_plan()` — The Main Engine

This is the primary allocation function, handling expiry risk assessment and the expiry-rescue transfer optimisation.

#### Phase 1: Load Inputs and Enrich Inventory

```python
as_of = pd.Timestamp(CFG["project"]["as_of_date"])
fcst  = load_forecasts()
repl  = load_replenishment()
mu    = fcst.groupby(["sku_id", "region"])["p50"].mean()

inv["days_to_expiry"]        = (inv["expiry_date"] - as_of).dt.days
inv["mu_here"]               = [mu.get((s, l), 0.0) for s, l in zip(inv["sku_id"], inv["location"])]
inv["sellable_before_expiry"] = np.minimum(inv["qty_units"], inv["mu_here"] * inv["days_to_expiry"].clip(lower=0))
inv["leftover"]              = (inv["qty_units"] - inv["sellable_before_expiry"]).clip(lower=0)
inv["risk_value_inr"]        = inv["leftover"] * inv["unit_cost_inr"]
```

Four columns are computed for each inventory batch:

| Column | Formula | Meaning |
|---|---|---|
| `days_to_expiry` | `expiry_date − as_of` | How many days until this batch expires |
| `mu_here` | `P50 mean demand at this location` | Expected daily sell-through rate for this SKU at this location |
| `sellable_before_expiry` | `min(qty, mu × days_to_expiry)` | Optimistic estimate: how much can be sold before expiry at the current location |
| `leftover` | `qty − sellable` (floored at 0) | Units **predicted to expire unsold** if no action is taken |
| `risk_value_inr` | `leftover × unit_cost` | Financial exposure from this batch (₹) |

> **Key insight:** `sellable_before_expiry` is capped at `qty_units` — you can't sell more than you have. `days_to_expiry` is also clipped at 0 so already-expired batches don't produce negative sellable figures.

#### Phase 2: Identify At-Risk Batches

```python
at_risk = inv[(inv["leftover"] > 5) & (inv["days_to_expiry"] <= 180)].copy()
```

A batch is classified as **at-risk** if:
- It has more than **5 units** predicted to expire unsold (`leftover > 5` — below this, action is not worth the logistics cost).
- Its expiry is within **180 days** — the horizon beyond which forecasts are unreliable and early intervention is premature.

Batches are sorted by `risk_value_inr` descending — highest financial exposure is addressed first.

#### Phase 3: Expiry Rescue — Finding the Best Destination

For each at-risk batch, the system searches all valid transfer destinations and scores them:

```python
for dest, lead_d in candidates:
    key = (b["sku_id"], dest)
    gap  = max(0.0, row["target_position"] - row["on_hand"])
    dest_dos = row["on_hand"] / mu_dest
    if dest_dos >= src_dos:         # don't transfer to somewhere with MORE supply
        continue
    consumable_before_expiry = mu_dest * b["days_to_expiry"]
    usable = min(remaining_leftover, max(gap, consumable_before_expiry * 0.8))
    if usable > 5:
        scored.append({..., "score": usable * b["unit_cost_inr"]})
```

**Eligibility filters for each candidate destination:**
- Must have this SKU in the replenishment plan's index (`key not in need.index` → skip).
- `dest_dos >= src_dos` → skip — do **not** transfer stock *to* somewhere that already has *more* days of supply than the source batch location. Only transfer to locations that need it more urgently.

**Quantity sizing (`usable`):**
```
usable = min(remaining_leftover, max(gap, consumable_before_expiry × 0.8))
```
- `remaining_leftover`: Units left in this batch not yet allocated.
- `gap`: The destination's replenishment deficit (target − on_hand).
- `consumable_before_expiry × 0.8`: How much the destination can realistically consume before the batch expires, with a 20% safety discount (to avoid sending more than can be sold).
- The `max(gap, consumable × 0.8)` ensures the transfer is sized generously enough to be worthwhile — the larger of the replenishment need or what can actually be consumed.

**Scoring:**
```
score = usable × unit_cost_inr
```
Candidates are ranked by financial value saved — prioritising high-value, large-volume transfers. This greedy scoring ensures the most financially impactful moves happen first.

**Allocation loop:**
```python
for c in scored:
    if remaining_leftover <= 5:
        break
    qty = int(min(c["usable"], remaining_leftover))
    ...
    remaining_leftover -= qty
```

The system allocates to destinations in score-descending order, decrementing `remaining_leftover` after each transfer. Once fewer than 5 units remain, it stops (below the minimum viable transfer size).

**Output fields:**
- `reason: "expiry_rescue"` — distinguishes these from shortage rescues.
- `value_saved_inr = qty × unit_cost_inr` — financial value rescued from write-off.
- `batch_id` — linked to the specific expiring batch for full traceability.

#### Phase 4: Merge Expiry and Shortage Rescues

```python
rescues = shortage_rescue_transfers(inv, repl, mu)
transfer_df = pd.concat([transfer_df, rescues], ignore_index=True)
```

The two transfer lists (expiry rescues and shortage rescues) are concatenated into a single `transfer_df`. Each row has a `reason` field (`"expiry_rescue"` or `"shortage_rescue"`) to maintain auditability.

#### Phase 5: Residual Write-Off Calculation

After allocating transfers, the system computes what write-off risk **remains**:

```python
transferred_by_batch = transfer_df.groupby("batch_id")["qty_units"].sum()

def residual(row):
    moved = int(transferred_by_batch.get(row["batch_id"], 0))
    return max(0.0, row["leftover"] - moved)

at_risk["residual_writeoff_units"] = at_risk.apply(residual, axis=1)
at_risk["residual_value_inr"]      = at_risk["residual_writeoff_units"] * at_risk["unit_cost_inr"]
```

For each at-risk batch, the residual write-off is: `leftover − qty_transferred_from_this_batch`. This produces the `writeoffs` table — all batches that still have projected write-off exposure even after the best transfers have been made. This is the **irreducible risk** that must be escalated for manual intervention or markdown pricing.

#### Phase 6: FEFO Order Generation

```python
fefo_order = inv.sort_values(["location", "sku_id", "expiry_date"])[
    ["location", "sku_id", "batch_id", "expiry_date", "qty_units", "status"]
]
```

**First-Expired, First-Out (FEFO)** — the pharmaceutical industry standard for dispensing/picking. Sorts the entire inventory by `expiry_date` ascending per location and SKU, providing a reference picking order for warehouse operations. This ensures that the oldest batches are always consumed first, minimising natural expiry under normal operations.

---

## `main()` — Execution and Output

```python
def main():
    transfers, writeoffs, _ = build_allocation_plan()
    db_out.write_transfer_plan(transfers, as_of)
    db_out.write_writeoff_risk(writeoffs, as_of)

    print(f"transfers planned: {len(transfers)} | units moved: {transfers.qty_units.sum():,.0f}")
    print(f"value rescued: ₹{transfers.value_saved_inr.sum():,.0f}")
    print(f"predicted residual write-offs after action: ₹{writeoffs.residual_value_inr.sum():,.0f}")
```

1. Builds the full allocation plan (expiry rescues + shortage rescues).
2. Writes to two PostgreSQL tables: `transfer_plan` and `writeoff_risk`.
3. Prints a console summary with three headline KPIs:
   - Total transfers planned and units moved.
   - Total financial value rescued from expiry (₹).
   - Residual write-off exposure that could not be rescued (₹).

---

## Output Tables

### `transfer_df` (→ `transfer_plan` in PostgreSQL)

| Column | Type | Description |
|---|---|---|
| `batch_id` | str / None | Source inventory batch ID (None for shortage rescues) |
| `sku_id` | str | Product identifier |
| `from_location` | str | Source warehouse/DC/region |
| `to_location` | str | Destination region |
| `qty_units` | int | Units to transfer |
| `expiry_date` | date / NaT | Batch expiry date (NaT for shortage rescues) |
| `days_to_expiry` | int / nan | Days until expiry from `as_of` |
| `transfer_lead_days` | int | Days for the physical transfer to arrive |
| `value_saved_inr` | int | Financial value rescued (₹) |
| `reason` | str | `"expiry_rescue"` or `"shortage_rescue"` |
| `src_days_of_supply_before` | float | Source DC's DOS before the transfer (shortage rescues only) |

### `writeoffs` (→ `writeoff_risk` in PostgreSQL)

| Column | Type | Description |
|---|---|---|
| `batch_id` | str | Inventory batch ID |
| `sku_id` | str | Product identifier |
| `location` | str | Current location of the batch |
| `qty_units` | int | Total units in the batch |
| `leftover` | float | Projected unsold units before transfers |
| `residual_writeoff_units` | float | Projected unsold units **after** all transfers |
| `unit_cost_inr` | int | Unit cost (₹) |
| `residual_value_inr` | float | Remaining write-off exposure (₹) |
| `expiry_date` | date | Batch expiry date |
| `days_to_expiry` | int | Days to expiry from `as_of` |

---

## Key Design Decisions

| Design Choice | Rationale |
|---|---|
| **Two separate rescue mechanisms** | Expiry rescue is *proactive* (moving surplus to prevent waste); shortage rescue is *reactive* (moving stock to prevent stockouts). Both are needed; neither alone is sufficient. |
| **`dest_dos >= src_dos` gate** | Prevents "stealing from the rich" transfers that worsen overall system balance. Stock only flows from higher-supply to lower-supply locations. |
| **`max(gap, consumable × 0.8)` sizing** | Ensures the transfer is large enough to be practically useful. Sending only the gap amount might under-transfer if the destination's gap is smaller than what can actually be consumed before expiry. |
| **Greedy scoring by `usable × unit_cost`** | Maximises total financial value rescued in a single pass. A globally optimal solution would require integer programming; the greedy approach is an excellent approximation at scale. |
| **5-unit minimum transfer threshold** | Below 5 units, the administrative and logistics cost of a transfer outweighs the financial saving. |
| **180-day at-risk window** | Forecasts beyond 6 months are unreliable. Intervening too early based on uncertain projections risks unnecessary logistics cost. |
| **DC-only sources for shortage rescue** | Prevents retail-to-retail lateral transfers that could cause cascading shortages across the network. DCs are the authorised redistribution nodes. |
| **FEFO output** | A pharmaceutical regulatory and quality management requirement — ensures the oldest stock is always dispensed first to minimise natural expiry under normal operations. |
