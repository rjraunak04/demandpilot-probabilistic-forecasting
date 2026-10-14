# Daily SKU Demand Panel

## Purpose

DemandPilot forecasting models operate on regularly defined demand
observations rather than individual retail transactions.

The daily demand aggregation stage converts forecasting-eligible transaction
rows into observed SKU-day demand.

## Forecasting Grain

The initial forecasting grain is:

`SKU x Day`

The target variable is:

`demand = total positive purchased units for a SKU on a calendar day`

Multiple transactions for the same SKU on the same day are therefore summed
into one observation.

## Demand Preservation

Daily aggregation must preserve total purchased units.

The pipeline checks:

`sum(transaction Quantity) == sum(daily demand)`

A mismatch is treated as a pipeline failure.

## Unique Grain

Every output row must uniquely represent one:

`sku + date`

Duplicate SKU-day observations are not permitted after aggregation.

## Missing Days

This stage only represents days on which positive demand was observed.

A missing SKU-day is not automatically converted to zero during aggregation.

Continuous SKU calendars and explicit zero-demand observations are introduced
in the next pipeline stage, where product eligibility boundaries can be
handled explicitly.

This separation prevents an absent transaction row from being silently
interpreted as zero demand without a defined business rule.

## Leakage

Daily aggregation uses only transactions occurring on the corresponding
calendar day.

No future demand information, rolling statistic, or model-derived quantity is
introduced at this stage.