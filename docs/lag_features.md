# Leakage-Safe Demand Lag Features

## Purpose

Demand forecasting requires information about historical demand without
exposing the model to the target value being predicted.

DemandPilot therefore creates lagged demand features independently within
each SKU time series.

## Lag Features

The initial feature set contains:

- `lag_1`
- `lag_7`
- `lag_14`
- `lag_28`

For a SKU on day `t`:

`lag_k(t) = demand(t - k)`

For example, `lag_7` represents demand for the same SKU seven calendar days
before the target date.

## Why the Continuous Panel Comes First

Lag offsets only have a consistent calendar interpretation when every
eligible day exists in the SKU time series.

DemandPilot therefore reconstructs zero-demand days before calculating lags.

Without this step, a one-row shift would mean "previous transaction day"
rather than "previous calendar day".

## Leakage Prevention

Lag features are calculated with group-wise historical shifts.

The target value at day `t` is never used to construct its own lag features.

Features are also grouped by SKU, preventing demand history from one product
from leaking into another product.

Automated tests additionally mutate future demand and verify that historical
lag features remain unchanged.

## Missing Lag History

Early observations in an SKU lifecycle do not have all historical lags.

For example, the first 28 calendar days cannot have a complete `lag_28`
history.

These missing values are preserved rather than automatically replaced with
zero because:

`unknown history != observed zero demand`

The modeling pipeline will define an explicit strategy for incomplete
historical features.

## Production Assumption

Lag generation assumes the input panel contains a continuous daily calendar
within each SKU's eligible history.

The pipeline validates this assumption and fails if calendar gaps remain.

## Next Stage

Rolling demand statistics and calendar features are added after the basic lag
features have been validated.

Rolling statistics must also be calculated from past-only information.