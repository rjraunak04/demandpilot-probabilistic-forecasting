# Model Card — DemandPilot 7-day probabilistic demand forecast

## Intended use

The model estimates seven-day SKU demand uncertainty for retail planning. It is
designed for forecasting demonstrations, backtesting, inventory scenario
analysis, and engineering evaluation.

It is not an autonomous purchasing system and should not place orders without
validated inventory, lead-time, service-level, and business constraints.

## Data

Source: UCI Online Retail II transaction history, covering 2009-12-01 through
2011-12-09 in the ingested project data. Demand is reconstructed at SKU-day
grain after project cleaning and validation.

The historical dataset does not represent a current production retailer and
does not provide a trustworthy current on-hand inventory ledger.

## Target and validation

The operational target is aggregate demand for the seven days following each
forecast origin. Feature construction is leakage-safe: lag and rolling inputs
use past observations only.

Model selection uses chronological validation. The locked test window
2011-11-12 through 2011-12-09 is not used for model selection.

## Model

DemandPilot uses LightGBM quantile regression to estimate P10, P50, and P90.
P50 is the central demand estimate. P10-P90 communicates forecast uncertainty,
and P90 is used as a service-oriented target-stock scenario when actual on-hand
inventory is supplied.

## Verified validation evidence

Across four weekly validation origins, the Day-04 probabilistic evaluation
contains 11,427 SKU-origin forecasts. P50 WAPE is 0.601275 versus 0.772276 for
the seasonal seven-day baseline, approximately 22.1% relative improvement.
Empirical P10-P90 interval coverage is 84.44%.

These are validation results, not claims about future commercial performance.

## Risks and limitations

- Historical transaction data may not transfer to a different retailer or era.
- Promotions, price strategy, holidays, assortment changes, and supply
  constraints may require richer production features.
- PSI detects distribution shift; it does not diagnose root cause or guarantee
  forecast degradation.
- Quantile coverage can change under drift and must be re-evaluated.
- Inventory recommendations require real on-hand stock and business constraints.
- Human review is required before model promotion or replenishment action.

## Monitoring and governance

Numeric feature drift uses deterministic PSI thresholds: stable below 0.10,
watch from 0.10 to below 0.25, and drift at or above 0.25. Drift triggers
investigation before retraining. Model promotion should require fresh
chronological backtests and release quality gates.
