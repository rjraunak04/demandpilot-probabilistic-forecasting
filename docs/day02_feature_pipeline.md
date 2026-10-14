# Day-02: Leakage-Safe Forecasting Feature Pipeline

## Objective

Day-02 converts cleaned retail transactions into a validated, model-ready
daily SKU forecasting dataset.

The pipeline is designed around chronological forecasting rather than random
row-wise prediction.

## Pipeline

Raw retail transactions

→ forecasting transaction preparation

→ observed SKU-day demand aggregation

→ continuous SKU lifetime calendar

→ explicit zero-demand reconstruction

→ leakage-safe lag features

→ past-only rolling statistics

→ deterministic calendar features

→ feature-contract validation

→ reproducible Parquet artifacts

## Forecast Grain

The forecasting grain is:

`SKU x Day`

No artificial store identifier is introduced because the source dataset does
not provide a reliable store dimension.

## Zero-Demand Policy

Missing dates inside an SKU's observed lifetime are reconstructed as explicit
zero-demand days.

No pre-launch zeros are created.

No post-last-observed-sale zeros are created because product availability and
discontinuation are not directly observed.

## Historical Features

Lag features:

- 1 day
- 7 days
- 14 days
- 28 days

Rolling windows:

- 7 days
- 14 days
- 28 days

For every rolling window, DemandPilot creates:

- mean
- standard deviation
- minimum
- maximum

All rolling features are shifted before calculation so demand on day `t`
cannot enter historical features for day `t`.

## Calendar Features

The pipeline creates:

- day of week
- day of month
- ISO week of year
- month
- quarter
- weekend indicator

These variables are known from the forecast date and do not expose future
target values.

## Validation

Automated validation covers:

- unique SKU-day grain
- continuous eligible calendars
- non-negative demand
- deterministic pipeline execution
- shuffled-input invariance
- same-day target isolation
- future-mutation leakage protection
- cross-SKU isolation
- historical-window boundaries
- feature schema contracts
- persisted artifact validation

## Materialized Artifacts

The pipeline produces:

- `data/processed/daily_sku_demand.parquet`
- `data/processed/forecast_features.parquet`
- `reports/data_quality/feature_pipeline_summary.json`

Large generated Parquet datasets remain outside Git because they are
reproducible build artifacts. The compact JSON audit summary is versioned.

## Day-02 Data Summary

The production pipeline transforms:

- 1,067,371 raw transaction rows
- 1,032,348 prepared transaction rows
- 534,499 observed SKU-day rows
- 2,302,303 continuous SKU-day rows
- 1,767,804 reconstructed zero-demand days
- 4,984 unique SKUs
- 26 final dataset columns

The final panel has a 76.78% zero-demand share, indicating a highly
intermittent retail-demand forecasting problem.

94.26% of rows have complete history for the maximum 28-day feature window.

## Modeling Boundary

Day-02 ends with validated forecasting features.

Day-03 introduces chronological train/validation/test boundaries, forecasting
baselines, tree-based models, rolling-origin backtesting, forecasting metrics,
and experiment tracking.