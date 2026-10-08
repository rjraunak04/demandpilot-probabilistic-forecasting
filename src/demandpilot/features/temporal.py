"""Create leakage-safe rolling and calendar forecasting features."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

DEFAULT_WINDOWS: tuple[int, ...] = (7, 14, 28)


@dataclass(frozen=True)
class TemporalFeatureReport:
    """Audit information for temporal feature generation."""

    input_rows: int
    output_rows: int
    unique_skus: int
    rolling_windows: tuple[int, ...]
    rows_with_complete_rolling_history: int
    rows_without_complete_rolling_history: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "unique_skus": self.unique_skus,
            "rolling_windows": list(self.rolling_windows),
            "rows_with_complete_rolling_history": (
                self.rows_with_complete_rolling_history
            ),
            "rows_without_complete_rolling_history": (
                self.rows_without_complete_rolling_history
            ),
        }


def _validate_windows(
    windows: Sequence[int],
) -> tuple[int, ...]:
    """Validate and normalize rolling windows."""
    normalized = tuple(windows)

    if not normalized:
        raise ValueError(
            "At least one rolling window must be provided."
        )

    if any(
        isinstance(window, bool)
        or not isinstance(window, int)
        for window in normalized
    ):
        raise ValueError(
            "Rolling windows must be positive integers."
        )

    if any(window <= 0 for window in normalized):
        raise ValueError(
            "Rolling windows must be positive integers."
        )

    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "Rolling windows must be unique."
        )

    return tuple(sorted(normalized))


def _validate_feature_input(data: pd.DataFrame) -> None:
    """Validate assumptions required for temporal features."""
    required_columns = {
        "sku",
        "date",
        "demand",
    }
    missing = required_columns.difference(data.columns)

    if missing:
        raise ValueError(
            "Missing columns required for temporal features: "
            f"{sorted(missing)}"
        )

    if data.empty:
        raise ValueError(
            "Feature dataset must not be empty."
        )

    if data["sku"].isna().any():
        raise ValueError(
            "SKU must not contain missing values."
        )

    if data["date"].isna().any():
        raise ValueError(
            "Date must not contain missing values."
        )

    if data["demand"].isna().any():
        raise ValueError(
            "Demand must not contain missing values."
        )

    if data.duplicated(["sku", "date"]).any():
        raise ValueError(
            "Feature dataset must have unique SKU-day grain."
        )


def add_temporal_features(
    data: pd.DataFrame,
    windows: Sequence[int] = DEFAULT_WINDOWS,
) -> tuple[pd.DataFrame, TemporalFeatureReport]:
    """Add past-only rolling statistics and deterministic calendar features."""
    _validate_feature_input(data)
    rolling_windows = _validate_windows(windows)

    featured = data.copy()

    featured["sku"] = (
        featured["sku"]
        .astype("string")
        .str.strip()
    )
    featured["date"] = pd.to_datetime(
        featured["date"],
        errors="raise",
        format="mixed",
    ).dt.normalize()

    featured["demand"] = pd.to_numeric(
        featured["demand"],
        errors="raise",
    )

    if featured["sku"].eq("").any():
        raise ValueError(
            "SKU must not contain blank values."
        )

    if (featured["demand"] < 0).any():
        raise ValueError(
            "Demand must be non-negative."
        )

    if featured.duplicated(["sku", "date"]).any():
        raise ValueError(
            "Date normalization produced duplicate SKU-day rows."
        )

    featured = featured.sort_values(
        ["sku", "date"],
        kind="stable",
    ).reset_index(drop=True)

    date_diff = (
        featured.groupby(
            "sku",
            observed=True,
        )["date"]
        .diff()
    )

    invalid_gap = (
        date_diff.notna()
        & date_diff.ne(pd.Timedelta(days=1))
    )

    if invalid_gap.any():
        raise ValueError(
            "Temporal features require a continuous daily calendar "
            "within each SKU."
        )

    past_demand = (
        featured.groupby(
            "sku",
            sort=False,
            observed=True,
        )["demand"]
        .shift(1)
    )

    rolling_columns: list[str] = []

    for window in rolling_windows:
        grouped_past = past_demand.groupby(
            featured["sku"],
            sort=False,
            observed=True,
        )

        mean_column = f"rolling_mean_{window}"
        std_column = f"rolling_std_{window}"
        min_column = f"rolling_min_{window}"
        max_column = f"rolling_max_{window}"

        featured[mean_column] = (
            grouped_past
            .rolling(
                window=window,
                min_periods=window,
            )
            .mean()
            .reset_index(
                level=0,
                drop=True,
            )
            .sort_index()
        )

        featured[std_column] = (
            grouped_past
            .rolling(
                window=window,
                min_periods=window,
            )
            .std(ddof=0)
            .reset_index(
                level=0,
                drop=True,
            )
            .sort_index()
        )

        featured[min_column] = (
            grouped_past
            .rolling(
                window=window,
                min_periods=window,
            )
            .min()
            .reset_index(
                level=0,
                drop=True,
            )
            .sort_index()
        )

        featured[max_column] = (
            grouped_past
            .rolling(
                window=window,
                min_periods=window,
            )
            .max()
            .reset_index(
                level=0,
                drop=True,
            )
            .sort_index()
        )

        rolling_columns.extend(
            [
                mean_column,
                std_column,
                min_column,
                max_column,
            ]
        )

    featured["day_of_week"] = (
        featured["date"].dt.dayofweek.astype("int8")
    )
    featured["day_of_month"] = (
        featured["date"].dt.day.astype("int8")
    )
    featured["week_of_year"] = (
        featured["date"]
        .dt.isocalendar()
        .week
        .astype("int16")
    )
    featured["month"] = (
        featured["date"].dt.month.astype("int8")
    )
    featured["quarter"] = (
        featured["date"].dt.quarter.astype("int8")
    )
    featured["is_weekend"] = (
        featured["day_of_week"]
        .isin([5, 6])
        .astype("int8")
    )

    complete_history = (
        featured[rolling_columns]
        .notna()
        .all(axis=1)
    )

    report = TemporalFeatureReport(
        input_rows=int(len(data)),
        output_rows=int(len(featured)),
        unique_skus=int(featured["sku"].nunique()),
        rolling_windows=rolling_windows,
        rows_with_complete_rolling_history=int(
            complete_history.sum()
        ),
        rows_without_complete_rolling_history=int(
            (~complete_history).sum()
        ),
    )

    return featured, report