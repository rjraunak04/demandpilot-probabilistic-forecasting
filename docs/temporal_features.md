# Temporal Forecasting Features

## Purpose

DemandPilot augments lagged demand with past-only rolling statistics and
deterministic calendar variables.

These features provide short- and medium-term demand context while preserving
the temporal boundary required for forecasting.

## Rolling Features

For windows of 7, 14, and 28 days, DemandPilot creates:

- rolling mean
- rolling standard deviation
- rolling minimum
- rolling maximum

The resulting feature set contains 12 rolling variables.

## Leakage-Safe Calculation

Every rolling statistic is calculated after shifting demand by one day.

Conceptually:

`rolling_feature(t) = statistic(demand before t)`

and never:

`statistic(demand including t)`

For example:

`rolling_mean_7(t)`

contains demand from:

`t-7 through t-1`

The target `demand(t)` is excluded.

## Missing Historical Context

A rolling feature is left missing until its complete historical window exists.

For example, a 28-day rolling statistic requires 28 previous calendar days.

Missing historical context is not replaced with zero because unknown history
is different from observed zero demand.

## SKU Isolation

All temporal calculations are performed independently within each SKU.

Demand history from one product cannot enter another product's rolling
features.

## Calendar Features

DemandPilot also creates deterministic date-derived features:

- `day_of_week`
- `day_of_month`
- `week_of_year`
- `month`
- `quarter`
- `is_weekend`

Calendar features depend only on the forecast date and therefore do not use
future target information.

## Validation

Automated tests verify:

- rolling windows exclude current-day demand
- future demand mutation does not change historical features
- same-day target mutation does not change same-day rolling features
- SKU histories remain isolated
- calendars remain continuous
- row counts remain unchanged
- input data is not mutated

## Modeling Boundary

Feature generation does not perform train/test splitting or model fitting.

Chronological training, validation, test boundaries and rolling-origin
backtesting are introduced in the modeling pipeline.