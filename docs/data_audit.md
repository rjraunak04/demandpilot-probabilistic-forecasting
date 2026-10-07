# DemandPilot Data Audit

## Dataset

DemandPilot uses the UCI Online Retail II transactional dataset.

The ingestion pipeline combines two yearly worksheets while preserving the
source worksheet for data provenance.

The raw dataset contains:

- 1,067,371 transaction rows
- 53,628 unique invoices
- 5,305 unique SKUs
- 5,942 unique customers
- 43 countries
- 604 active calendar days

The raw records represent transaction events rather than forecasting
observations. A downstream transformation stage will convert eligible
transactions into regular daily SKU-level demand series.

## Audit Scope

The Day-01 profiling workflow evaluates:

- temporal coverage
- invoice, SKU, customer, and country cardinality
- missing values
- negative and zero quantities
- negative and zero prices
- cancellations and returns
- exact duplicate rows
- country distribution
- SKU frequency
- source-sheet coverage

Profiling is intentionally non-mutating. No cleaning, filtering, imputation,
aggregation, or forecasting transformation is performed during this stage.

## Key Audit Findings

The raw-data audit identified:

- 22,950 rows with negative quantities
- 6,202 rows with zero prices
- 19,494 cancellation-invoice rows
- 12,133 exact duplicate rows
- zero rows with zero quantity

These observations are retained during profiling because the audit stage is
responsible for measuring data characteristics rather than silently modifying
the source data.

## Key Modeling Challenges

### 1. Returns and cancellations

Negative quantities and cancellation invoices represent business events that
cannot automatically be interpreted as ordinary customer demand.

Demand construction must therefore define an explicit and reproducible rule
for returns and cancellations before model training.

### 2. Transaction data is not a regular time series

The source dataset records purchase events at transaction timestamps.

Forecasting requires these events to be transformed into a regular daily
SKU-level demand grid. The transformation must also represent eligible dates
on which no sales were observed so that absence of a transaction is not
confused with missing data.

### 3. Data quality and sparse product histories

Zero-price transactions, duplicate rows, missing values, and irregular SKU
sales histories require explicit downstream treatment.

Data-quality decisions must be separated from forecasting transformations so
that assumptions remain auditable and reproducible.

## Leakage Consideration

Demand forecasting is inherently temporal.

All downstream aggregation, feature engineering, train-validation-test
splitting, backtesting, and model evaluation must preserve chronological
order. Future observations must never contribute information to features used
for predicting earlier dates.

## Reproducible Audit

The executable profiling workflow is implemented in:

`src/demandpilot/data/profile.py`

The audit can be reproduced with:

`python scripts/run_data_audit.py`

Measured statistics are written to:

`reports/data_quality/data_quality_summary.json`

The JSON artifact remains the machine-readable source of truth for the
measured data-quality statistics.