"""Validation contracts for model-ready forecasting features."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from demandpilot.features.lags import DEFAULT_LAGS
from demandpilot.features.temporal import DEFAULT_WINDOWS

CALENDAR_COLUMNS: tuple[str, ...] = (
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "is_weekend",
)


def expected_feature_columns() -> set[str]:
    """Return columns required from the Day-02 feature pipeline."""
    lag_columns = {
        f"lag_{lag}"
        for lag in DEFAULT_LAGS
    }

    rolling_columns = {
        f"rolling_{statistic}_{window}"
        for window in DEFAULT_WINDOWS
        for statistic in (
            "mean",
            "std",
            "min",
            "max",
        )
    }

    return {
        "sku",
        "date",
        "demand",
        "is_observed_demand",
        *lag_columns,
        *rolling_columns,
        *CALENDAR_COLUMNS,
    }


def validate_forecasting_features(
    features: pd.DataFrame,
) -> None:
    """Enforce Day-02 forecasting feature invariants."""
    if features.empty:
        raise ValueError(
            "Forecasting feature dataset must not be empty."
        )

    required = expected_feature_columns()
    missing = required.difference(features.columns)

    if missing:
        raise ValueError(
            "Forecasting feature dataset is missing columns: "
            f"{sorted(missing)}"
        )

    if features["sku"].isna().any():
        raise ValueError(
            "Feature dataset contains missing SKU values."
        )

    if features["date"].isna().any():
        raise ValueError(
            "Feature dataset contains missing dates."
        )

    if features["demand"].isna().any():
        raise ValueError(
            "Feature dataset contains missing target values."
        )

    if features.duplicated(["sku", "date"]).any():
        raise ValueError(
            "Feature dataset contains duplicate SKU-day rows."
        )

    if (features["demand"] < 0).any():
        raise ValueError(
            "Feature dataset contains negative demand."
        )

    if not features["is_observed_demand"].isin(
        [True, False]
    ).all():
        raise ValueError(
            "is_observed_demand must be boolean."
        )

    if not features["day_of_week"].between(0, 6).all():
        raise ValueError(
            "day_of_week is outside the expected range."
        )

    if not features["day_of_month"].between(1, 31).all():
        raise ValueError(
            "day_of_month is outside the expected range."
        )

    if not features["week_of_year"].between(1, 53).all():
        raise ValueError(
            "week_of_year is outside the expected range."
        )

    if not features["month"].between(1, 12).all():
        raise ValueError(
            "month is outside the expected range."
        )

    if not features["quarter"].between(1, 4).all():
        raise ValueError(
            "quarter is outside the expected range."
        )

    if not features["is_weekend"].isin([0, 1]).all():
        raise ValueError(
            "is_weekend must contain only 0 or 1."
        )

    sorted_features = features.sort_values(
        ["sku", "date"],
        kind="stable",
    ).reset_index(drop=True)

    date_diff = (
        sorted_features.groupby(
            "sku",
            observed=True,
        )["date"]
        .diff()
    )

    invalid_gap = (
        date_diff.notna()
        & date_diff.ne(timedelta(days=1))
    )

    if invalid_gap.any():
        raise ValueError(
            "Feature dataset contains calendar gaps within SKU history."
        )