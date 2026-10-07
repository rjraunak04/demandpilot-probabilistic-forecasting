# DemandPilot Testing Strategy

## Objective

DemandPilot uses automated tests to verify that data ingestion and validation
behavior remains deterministic, reproducible, and safe for downstream
forecasting.

## Testing Approach

Unit tests use small synthetic datasets rather than repeatedly loading the
full Online Retail II workbook.

This provides:

- fast execution
- deterministic behavior
- explicit edge-case coverage
- CI-friendly quality gates
- isolation from the large raw dataset

## Validation Coverage

Automated tests verify that the data-contract layer rejects:

- empty datasets
- missing required columns
- missing core identifiers
- non-numeric quantities
- non-numeric prices
- malformed invoice dates
- unexpected source worksheets
- missing source provenance

Tests also verify that legitimate or business-dependent conditions are
observed without being incorrectly rejected, including:

- negative quantities
- cancellation invoices
- zero-price transactions
- missing customer identifiers
- missing product descriptions

## Ingestion Coverage

File-level tests verify that:

- missing raw workbooks fail loudly
- unsupported file extensions are rejected
- configured Excel workbooks are accepted
- ingestion metadata summaries remain deterministic

## Quality Gates

Before a checkpoint is committed, the repository must pass:

```bash
ruff check .
python -m compileall -q src scripts
pytest -q