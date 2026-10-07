# DemandPilot Data Contracts

## Purpose

DemandPilot validates structural assumptions before retail transactions enter
downstream forecasting transformations.

Validation is intentionally separated from cleaning. The contract layer fails
on structurally unusable data while preserving legitimate or ambiguous
business events for explicit downstream treatment.

## Enforced Contracts

The raw retail dataset must:

1. be non-empty
2. contain all required columns
3. contain Invoice, StockCode, InvoiceDate, and Country identifiers
4. contain numeric Quantity values
5. contain numeric Price values
6. contain parseable InvoiceDate values
7. retain valid source-workbook provenance

## Validation Result

The Online Retail II dataset passed all 7 enforced structural contracts across
1,067,371 transaction rows and 9 columns.

The validator also recorded the following business and data-quality conditions:

- 22,950 negative-quantity rows
- 0 zero-quantity rows
- 5 negative-price rows
- 6,202 zero-price rows
- 19,494 cancellation-invoice rows
- 12,133 exact duplicate rows
- 4,382 rows with missing product descriptions
- 243,007 rows with missing customer identifiers

## Observed but Not Rejected

The validator records but does not automatically reject negative quantities,
zero or negative prices, cancellations, duplicate rows, missing descriptions,
or missing customer identifiers.

These observations require business-aware treatment during demand construction
rather than silent modification during raw-data validation.

## Design Principle

A data contract answers:

> Is this dataset structurally usable by the pipeline?

It does not answer:

> Which transactions should contribute to forecastable demand?

The second question belongs to downstream demand-definition and transformation
logic.

## Leakage

The validation layer performs no temporal aggregation, imputation, feature
engineering, target construction, or train-test splitting. Therefore, it does
not introduce future information into historical observations.