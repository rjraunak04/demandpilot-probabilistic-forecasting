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


## Verified validation results

The Day-04 pipeline was verified across four weekly fixed-origin forecasts with
the locked test still untouched.

| Origin | Forecast rows | P50 WAPE | Seasonal 7d WAPE | 80% coverage | Mean interval width |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2011-10-15 | 2,888 | 0.594846 | 0.704362 | 84.14% | 86.05 |
| 2011-10-22 | 2,883 | 0.624645 | 0.905586 | 84.70% | 71.23 |
| 2011-10-29 | 2,846 | 0.614509 | 0.778047 | 84.61% | 83.65 |
| 2011-11-05 | 2,810 | 0.571101 | 0.701109 | 84.31% | 104.94 |

Across 11,427 SKU-origin forecasts, mean P50 WAPE was **0.601275** versus
**0.772276** for the previous-seven-day seasonal total baseline, a relative
improvement of approximately **22.1%**. Mean empirical coverage of the nominal
P10-P90 interval was **84.44%**.

These are validation results, not locked-test claims. Reorder quantities remain
scenario-only until real on-hand inventory is supplied.
