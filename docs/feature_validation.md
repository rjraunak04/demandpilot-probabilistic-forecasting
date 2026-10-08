# Forecasting Feature Validation

## Purpose

DemandPilot treats feature engineering as a production data pipeline rather
than a notebook-only transformation.

The Day-02 validation layer verifies that the final forecasting feature
dataset is structurally valid, temporally consistent, deterministic, and
protected against common forms of target leakage.

## End-to-End Pipeline

The feature pipeline composes:

1. raw transaction validation and forecasting preparation
2. SKU-day demand aggregation
3. zero-demand calendar reconstruction
4. leakage-safe lag features
5. past-only rolling statistics
6. deterministic calendar features
7. final feature-contract validation

## Feature Contract

Every output row represents one unique:

`SKU x Day`

Required fields include:

- target demand
- observed/reconstructed demand indicator
- lag features
- rolling demand statistics
- calendar features

Historical lag and rolling values may be missing when sufficient past history
does not yet exist.

These values are intentionally not replaced with zero.

## Leakage Guardrails

Rolling statistics use shifted historical demand so demand for day `t` cannot
enter historical features for day `t`.

Automated mutation tests alter future demand and verify that historical
features remain unchanged.

Feature calculations are isolated by SKU. Changing one SKU must not alter
another SKU's feature history.

## Calendar Integrity

Each eligible SKU history contains one row per calendar day between its
defined first and last observed-demand boundaries.

Lag offsets therefore represent calendar-day offsets rather than previous
transaction rows.

## Determinism

Repeated execution on identical input must produce identical output.

The pipeline is also tested against shuffled raw transaction ordering so the
final feature dataset does not depend on incidental source-row order.

## Data Preservation

Feature engineering preserves:

- unique SKU-day grain
- non-negative demand
- continuous eligible SKU calendars
- panel row count
- deterministic temporal ordering

## Modeling Boundary

Day-02 creates and validates forecasting features only.

Chronological train, validation, locked-test boundaries and rolling-origin
backtesting are introduced separately during model development.