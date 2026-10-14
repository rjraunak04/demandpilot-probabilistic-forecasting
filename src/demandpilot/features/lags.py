"""Create leakage-safe lagged demand features."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pandas as pd

DEFAULT_LAGS: tuple[int, ...] = (1, 7, 14, 28)


@dataclass(frozen=True)
class LagFeatureReport:
    """Audit information for lag-feature generation."""

    input_rows: int
    output_rows: int
    unique_skus: int
    lag_days: tuple[int, ...]
    rows_with_complete_lag_history: int
    rows_without_complete_lag_history: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "unique_skus": self.unique_skus,
            "lag_days": list(self.lag_days),
            "rows_with_complete_lag_history": (
                self.rows_with_complete_lag_history
            ),
            "rows_without_complete_lag_history": (
                self.rows_without_complete_lag_history
            ),
        }


def _validate_lags(lags: Sequence[int]) -> tuple[int, ...]:
    """Validate and normalize requested lag periods."""
    normalized = tuple(lags)

    if not normalized:
        raise ValueError("At least one lag must be provided.")

    if any(
        isinstance(lag, bool)
        or not isinstance(lag, int)
        for lag in normalized
    ):
        raise ValueError(
            "Lag values must be positive integers."
        )

    if any(lag <= 0 for lag in normalized):
        raise ValueError(
            "Lag values must be positive integers."
        )

    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "Lag values must be unique."
        )

    return tuple(sorted(normalized))


def _validate_panel(panel: pd.DataFrame) -> None:
    """Validate assumptions required for lag generation."""
    required_columns = {
        "sku",
        "date",
        "demand",
    }
    missing = required_columns.difference(panel.columns)

    if missing:
        raise ValueError(
            f"Missing columns required for lag features: "
            f"{sorted(missing)}"
        )

    if panel.empty:
        raise ValueError(
            "Demand panel must not be empty."
        )

    if panel["sku"].isna().any():
        raise ValueError(
            "SKU must not contain missing values."
        )

    if panel["date"].isna().any():
        raise ValueError(
            "Date must not contain missing values."
        )

    if panel["demand"].isna().any():
        raise ValueError(
            "Demand must not contain missing values."
        )

    if panel.duplicated(["sku", "date"]).any():
        raise ValueError(
            "Demand panel must have unique SKU-day grain."
        )


def add_demand_lags(
    panel: pd.DataFrame,
    lags: Sequence[int] = DEFAULT_LAGS,
) -> tuple[pd.DataFrame, LagFeatureReport]:
    """Add past-only demand lags independently within each SKU.

    Lag features are generated with group-wise ``shift``. Therefore the
    feature for date t contains demand from an earlier date and never demand
    from t itself.

    The function expects a continuous daily panel within each SKU.
    """
    _validate_panel(panel)
    lag_days = _validate_lags(lags)

    data = panel.copy()

    data["sku"] = (
        data["sku"]
        .astype("string")
        .str.strip()
    )
    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
        format="mixed",
    ).dt.normalize()

    data["demand"] = pd.to_numeric(
        data["demand"],
        errors="raise",
    )

    if data["sku"].eq("").any():
        raise ValueError(
            "SKU must not contain blank values."
        )

    if (data["demand"] < 0).any():
        raise ValueError(
            "Demand must be non-negative."
        )

    if data.duplicated(["sku", "date"]).any():
        raise ValueError(
            "Date normalization produced duplicate SKU-day rows."
        )

    data = data.sort_values(
        ["sku", "date"],
        kind="stable",
    ).reset_index(drop=True)

    date_diff = (
        data.groupby(
            "sku",
            observed=True,
        )["date"]
        .diff()
    )

    expected_step = timedelta(days=1)

    invalid_gap = (
        date_diff.notna()
        & date_diff.ne(expected_step)
    )

    if invalid_gap.any():
        raise ValueError(
            "Lag generation requires a continuous daily calendar "
            "within each SKU."
        )

    grouped_demand = data.groupby(
        "sku",
        sort=False,
        observed=True,
    )["demand"]

    lag_columns: list[str] = []

    for lag in lag_days:
        column = f"lag_{lag}"
        data[column] = grouped_demand.shift(lag)
        lag_columns.append(column)

    complete_history = data[lag_columns].notna().all(axis=1)

    report = LagFeatureReport(
        input_rows=int(len(panel)),
        output_rows=int(len(data)),
        unique_skus=int(data["sku"].nunique()),
        lag_days=lag_days,
        rows_with_complete_lag_history=int(
            complete_history.sum()
        ),
        rows_without_complete_lag_history=int(
            (~complete_history).sum()
        ),
    )

    return data, report