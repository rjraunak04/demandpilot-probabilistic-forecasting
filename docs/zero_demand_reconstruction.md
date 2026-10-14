# Zero-Demand Calendar Reconstruction

## Purpose

Retail transaction data records purchases rather than explicit daily demand
observations.

A day without a transaction is therefore absent from the source data.

Forecasting models require a clearly defined time index, so DemandPilot
reconstructs missing eligible SKU-days explicitly.

## Eligibility Boundary

For the initial forecasting design, each SKU is considered active from:

`first observed positive-demand date`

through:

`last observed positive-demand date`

Only missing dates inside this observed lifetime are reconstructed.

DemandPilot does not create zeros before the first observed sale because
those dates may precede product introduction.

DemandPilot also does not create zeros after the final observed sale because
the source data does not establish whether the SKU remained available or was
discontinued.

## Zero-Demand Definition

Within an SKU's observed lifetime:

- observed positive-demand day -> original demand is preserved
- missing calendar day -> demand is set to zero

Each reconstructed row also contains `is_observed_demand`.

This field distinguishes source-observed positive demand from calendar rows
introduced by the forecasting pipeline.

## Demand Preservation

Calendar reconstruction must not change total demand.

The pipeline verifies:

`sum(observed demand) == sum(reconstructed panel demand)`

Inserted rows contribute zero units.

## Forecasting Grain

After reconstruction, each row uniquely represents:

`SKU x Day`

Each SKU calendar must be continuous between its first and last observed
positive-demand dates.

## Leakage

Calendar reconstruction does not use future demand values to calculate
features.

The last observed positive-demand date is currently used only to define the
retrospective dataset boundary for panel construction.

For production inference, active assortment and availability information
would be preferable for determining whether future zero-demand observations
represent true demand opportunities.

Lagged and rolling predictors are introduced in later stages and must use
past-only demand information.