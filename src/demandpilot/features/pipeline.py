"""Compose the DemandPilot forecasting feature pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from demandpilot.features.daily import build_daily_demand
from demandpilot.features.lags import DEFAULT_LAGS, add_demand_lags
from demandpilot.features.panel import reconstruct_zero_demand_days
from demandpilot.features.prepare import prepare_forecasting_transactions
from demandpilot.features.temporal import (
    DEFAULT_WINDOWS,
    add_temporal_features,
)


@dataclass(frozen=True)
class FeaturePipelineReport:
    """End-to-end forecasting feature pipeline audit."""

    raw_rows: int
    prepared_rows: int
    observed_daily_rows: int
    panel_rows: int
    inserted_zero_rows: int
    feature_rows: int
    unique_skus: int
    lag_days: tuple[int, ...]
    rolling_windows: tuple[int, ...]
    complete_history_rows: int
    incomplete_history_rows: int

    @property
    def complete_history_share(self) -> float:
        """Return the share of rows with complete maximum history."""
        if self.feature_rows == 0:
            return 0.0

        return self.complete_history_rows / self.feature_rows

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "raw_rows": self.raw_rows,
            "prepared_rows": self.prepared_rows,
            "observed_daily_rows": self.observed_daily_rows,
            "panel_rows": self.panel_rows,
            "inserted_zero_rows": self.inserted_zero_rows,
            "feature_rows": self.feature_rows,
            "unique_skus": self.unique_skus,
            "lag_days": list(self.lag_days),
            "rolling_windows": list(self.rolling_windows),
            "complete_history_rows": self.complete_history_rows,
            "incomplete_history_rows": self.incomplete_history_rows,
            "complete_history_share": self.complete_history_share,
        }


def build_forecasting_features(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, FeaturePipelineReport]:
    """Build the complete leakage-safe forecasting feature dataset."""
    prepared, preparation_report = prepare_forecasting_transactions(
        raw
    )

    daily, daily_report = build_daily_demand(
        prepared
    )

    panel, panel_report = reconstruct_zero_demand_days(
        daily
    )

    lagged, lag_report = add_demand_lags(
        panel
    )

    featured, temporal_report = add_temporal_features(
        lagged
    )

    lag_columns = [
        f"lag_{lag}"
        for lag in DEFAULT_LAGS
    ]

    rolling_columns = [
        f"rolling_{statistic}_{window}"
        for window in DEFAULT_WINDOWS
        for statistic in (
            "mean",
            "std",
            "min",
            "max",
        )
    ]

    history_columns = lag_columns + rolling_columns

    complete_history = (
        featured[history_columns]
        .notna()
        .all(axis=1)
    )

    if len(featured) != len(panel):
        raise RuntimeError(
            "Feature engineering changed panel row count."
        )

    if featured.duplicated(["sku", "date"]).any():
        raise RuntimeError(
            "Feature pipeline produced duplicate SKU-day rows."
        )

    if (
        lag_report.output_rows
        != temporal_report.output_rows
    ):
        raise RuntimeError(
            "Lag and temporal feature row counts do not reconcile."
        )

    report = FeaturePipelineReport(
        raw_rows=int(len(raw)),
        prepared_rows=preparation_report.output_rows,
        observed_daily_rows=daily_report.output_daily_rows,
        panel_rows=panel_report.panel_rows,
        inserted_zero_rows=panel_report.inserted_zero_rows,
        feature_rows=int(len(featured)),
        unique_skus=int(featured["sku"].nunique()),
        lag_days=DEFAULT_LAGS,
        rolling_windows=DEFAULT_WINDOWS,
        complete_history_rows=int(
            complete_history.sum()
        ),
        incomplete_history_rows=int(
            (~complete_history).sum()
        ),
    )

    return featured, report