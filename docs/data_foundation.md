# DemandPilot Data Foundation

## Purpose

The DemandPilot data foundation converts an external retail workbook into a
validated and auditable source suitable for downstream forecasting.

The foundation intentionally separates four responsibilities:

1. ingestion
2. profiling
3. structural validation
4. forecasting transformations

This prevents data cleaning assumptions from being silently embedded inside
the ingestion layer.

## Source Data

DemandPilot uses the UCI Online Retail II transaction dataset.

The raw workbook contains two historical worksheets. Both are ingested and
combined while retaining worksheet provenance through `SourceSheet`.

The raw dataset is treated as immutable input and is excluded from version
control.

## Ingestion Layer

`src/demandpilot/data/ingest.py`

Responsibilities:

- load configuration
- resolve dataset paths
- verify workbook availability
- load configured worksheets
- preserve source provenance
- combine transaction records
- expose lightweight ingestion metadata

The ingestion layer does not perform business cleaning.

## Profiling Layer

`src/demandpilot/data/profile.py`

The profiling workflow characterizes the raw dataset before modeling
decisions are made.

It measures transaction volume, entity cardinality, time coverage,
missingness, cancellations, returns, duplicates, and price/quantity
conditions.

Profiling is observational and does not mutate the raw dataset.

## Validation Layer

`src/demandpilot/data/validate.py`

Structural contracts reject data that would make downstream processing
unsafe or ambiguous.

Business-dependent anomalies are reported rather than automatically
discarded.

This distinction is important because a negative quantity may represent a
valid return, while a non-numeric quantity represents a structural data
failure.

## Testing Layer

Synthetic DataFrames are used for unit tests so that validation behavior can
be tested quickly without repeatedly loading the million-row workbook.

The complete raw dataset is additionally used by the verification script as
a regression check.

This provides both:

- fast deterministic unit testing
- real-dataset integration verification

## Forecasting Boundary

Raw transactions are not directly treated as forecasting observations.

Before training, DemandPilot must construct a regular chronological demand
panel at the SKU-day grain.

This transformation must explicitly represent zero-sales days and preserve
temporal ordering.

No future information may be used to construct historical model features.

## Day-01 Outcome

The data foundation establishes a reproducible boundary between raw retail
transactions and the future forecasting pipeline.

Forecasting development can now proceed without embedding ingestion,
validation, or source-quality assumptions directly inside model code.