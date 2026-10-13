"""Horizon-safe lead-time demand targets for inventory decisions."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from demandpilot.models.boosting import MODEL_FEATURES


def build_lead_time_examples(
    features: pd.DataFrame,
    *,
    horizon_days: int = 7,
) -> pd.DataFrame:
    """Build snapshot rows whose target is demand over [t, t+horizon-1].

    Features remain those available at the snapshot date t and are therefore
    based only on demand through t-1. Targets never cross SKU lifetime gaps.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")
    required = {"sku", "date", "demand", *MODEL_FEATURES}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Missing lead-time columns: {sorted(missing)}")
    if features.empty:
        raise ValueError("Feature dataset must not be empty.")

    data = features.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise").dt.normalize()
    data = data.sort_values(["sku", "date"], kind="stable").reset_index(drop=True)
    if data.duplicated(["sku", "date"]).any():
        raise ValueError("Lead-time input must have unique SKU-day grain.")

    grouped = data.groupby("sku", sort=False, observed=True)
    future_parts = [grouped["demand"].shift(-offset) for offset in range(horizon_days)]
    future = pd.concat(future_parts, axis=1)
    target_end_date = grouped["date"].shift(-(horizon_days - 1))
    expected_end = data["date"] + timedelta(days=horizon_days - 1)
    complete_window = target_end_date.eq(expected_end) & future.notna().all(axis=1)

    data["lead_time_demand"] = future.sum(axis=1, min_count=horizon_days)
    data["target_end"] = target_end_date
    return data.loc[complete_window].reset_index(drop=True)


def training_examples_at_origin(
    examples: pd.DataFrame,
    origin: str | pd.Timestamp,
) -> pd.DataFrame:
    """Return examples whose complete labels are known before an origin."""
    origin_ts = pd.Timestamp(origin)
    required = {"date", "target_end", "lead_time_demand"}
    missing = required.difference(examples.columns)
    if missing:
        raise ValueError(f"Missing lead-time example columns: {sorted(missing)}")
    train = examples.loc[examples["target_end"] < origin_ts].copy()
    train = train.dropna(subset=list(MODEL_FEATURES))
    if train.empty:
        raise ValueError("No completed lead-time labels are available before origin.")
    return train.reset_index(drop=True)


def snapshot_examples(
    examples: pd.DataFrame,
    origin: str | pd.Timestamp,
) -> pd.DataFrame:
    """Return the SKU snapshot predicted at a forecast origin."""
    origin_ts = pd.Timestamp(origin)
    snapshot = examples.loc[examples["date"] == origin_ts].copy()
    if snapshot.empty:
        raise ValueError(f"No lead-time examples at origin {origin_ts.date()}.")
    return snapshot.reset_index(drop=True)
