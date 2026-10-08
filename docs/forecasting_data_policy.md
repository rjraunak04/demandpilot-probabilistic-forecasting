# Forecasting Data Policy

## Purpose

DemandPilot separates raw transaction validation from forecasting-specific
business preparation.

Raw validation determines whether the source data is structurally usable.
Forecasting preparation determines which structurally valid transactions
represent positive realized unit demand.

## Positive Demand Policy

The initial demand target represents positive purchased units.

Before daily aggregation, the pipeline:

1. removes exact duplicate transaction rows
2. excludes cancellation invoices
3. excludes remaining non-positive quantities
4. requires product and timestamp identifiers
5. preserves zero and negative price records for auditability
6. preserves missing customer identifiers

Returns and cancellations are not converted into positive demand.

## Why Price Is Not a Demand Filter

DemandPilot forecasts unit demand rather than transaction revenue.

A zero or unusual price does not automatically prove that no product units
were demanded. Price-quality issues therefore remain auditable and can be
handled separately if later modeling requires price-based features.

## Forecasting Boundary

This stage does not create daily demand observations.

Daily SKU aggregation and zero-demand calendar reconstruction are separate
pipeline stages so their assumptions remain explicit and independently
testable.

## Leakage

Transaction preparation uses only information contained in each historical
transaction and does not calculate future-dependent statistics.

Lagged, rolling, and other temporal predictors are introduced only after the
daily demand panel has been constructed.