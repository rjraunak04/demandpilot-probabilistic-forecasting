# Day 04 — Probabilistic Forecasting and Inventory Intelligence

## Why the target changes

Day-03 daily-row benchmarks are retained as diagnostics. For operational
inventory decisions, Day 04 uses a stricter **7-day lead-time target**.

At forecast origin `t`, model features are the Day-02 lag/rolling features
known through `t-1`. The target is total SKU demand from `t` through
`t+6`. A training example is eligible only when its target window ends before
the next forecast origin. This prevents actual demand inside a forecast horizon
from entering its own prediction features.

This distinction makes the primary Day-04 evaluation genuinely fixed-origin
and inventory-aligned.

## Checkpoints

1. Build horizon-safe 7-day lead-time targets.
2. Fit LightGBM quantile regressors for P10/P50/P90.
3. Enforce quantile coherence by monotone rearrangement.
4. Evaluate pinball loss, P50 WAPE, 80% interval coverage, and interval width.
5. Run four weekly rolling forecast origins across the validation period.
6. Convert uncertainty into target-stock and safety-stock proxies.
7. Support stockout-risk and reorder decisions when real on-hand stock exists.
8. Materialize reproducible reports and validate the full implementation.

## Forecast interpretation

- **P10**: lower-demand scenario.
- **P50**: median lead-time demand forecast.
- **P90**: service-oriented high-demand scenario.
- **P90 - P50**: uncertainty-derived safety-stock proxy.
- **P90 - P10**: forecast uncertainty width.

P10/P50/P90 are independently trained quantile regressions. Quantile crossing
is corrected by sorting the three predictions per SKU forecast.

## Inventory honesty

Online Retail II contains transactions, not a reliable current on-hand
inventory ledger. DemandPilot therefore does **not fabricate stock levels**.

Without on-hand inventory, the pipeline reports target-stock requirements and
marks reorder quantities as `inventory_required`. When a real non-negative
on-hand vector is supplied:

- on-hand < P50 → high stockout risk
- P50 <= on-hand < P90 → medium risk
- on-hand >= P90 → low risk
- reorder quantity = max(P90 - on-hand, 0), subject to optional MOQ

These are transparent policy rules rather than claims of an optimized supply
chain cost function.

## Locked test policy

The locked test period remains untouched during Day-04 model development.
Probabilistic validation uses the four weekly origins inside the existing
validation window.
