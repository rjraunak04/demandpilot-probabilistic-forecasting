"""Verify temporal forecasting features on the complete retail dataset."""

from demandpilot.data.ingest import load_retail_data
from demandpilot.features.daily import build_daily_demand
from demandpilot.features.lags import add_demand_lags
from demandpilot.features.panel import reconstruct_zero_demand_days
from demandpilot.features.prepare import prepare_forecasting_transactions
from demandpilot.features.temporal import (
    DEFAULT_WINDOWS,
    add_temporal_features,
)


def main() -> None:
    """Run DemandPilot through rolling and calendar feature generation."""
    raw = load_retail_data()

    prepared, _ = prepare_forecasting_transactions(raw)
    daily, _ = build_daily_demand(prepared)
    panel, _ = reconstruct_zero_demand_days(daily)
    lagged, lag_report = add_demand_lags(panel)

    featured, temporal_report = add_temporal_features(
        lagged
    )

    assert len(featured) == len(lagged)
    assert not featured.duplicated(["sku", "date"]).any()

    assert featured["demand"].equals(
        lagged["demand"]
    )

    for window in DEFAULT_WINDOWS:
        for statistic in (
            "mean",
            "std",
            "min",
            "max",
        ):
            column = f"rolling_{statistic}_{window}"
            assert column in featured.columns

    calendar_columns = [
        "day_of_week",
        "day_of_month",
        "week_of_year",
        "month",
        "quarter",
        "is_weekend",
    ]

    for column in calendar_columns:
        assert column in featured.columns
        assert featured[column].notna().all()

    assert featured["day_of_week"].between(0, 6).all()
    assert featured["day_of_month"].between(1, 31).all()
    assert featured["month"].between(1, 12).all()
    assert featured["quarter"].between(1, 4).all()
    assert featured["is_weekend"].isin([0, 1]).all()

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

    complete_history = (
        featured[rolling_columns]
        .notna()
        .all(axis=1)
    )

    assert (
        int(complete_history.sum())
        == temporal_report.rows_with_complete_rolling_history
    )

    assert (
        temporal_report.rows_with_complete_rolling_history
        + temporal_report.rows_without_complete_rolling_history
        == len(featured)
    )

    assert (
        temporal_report.output_rows
        == lag_report.output_rows
    )

    print("DemandPilot CP-05 temporal features PASSED")
    print(
        f"Input rows: "
        f"{temporal_report.input_rows:,}"
    )
    print(
        f"Output rows: "
        f"{temporal_report.output_rows:,}"
    )
    print(
        f"Unique SKUs: "
        f"{temporal_report.unique_skus:,}"
    )
    print(
        "Rolling windows: "
        + ", ".join(
            str(window)
            for window in temporal_report.rolling_windows
        )
    )
    print(
        "Rolling feature count: "
        f"{len(rolling_columns)}"
    )
    print(
        "Calendar feature count: "
        f"{len(calendar_columns)}"
    )
    print(
        "Rows with complete rolling history: "
        f"{temporal_report.rows_with_complete_rolling_history:,}"
    )
    print(
        "Rows without complete rolling history: "
        f"{temporal_report.rows_without_complete_rolling_history:,}"
    )
    print(
        "Complete rolling-history share: "
        f"{temporal_report.rows_with_complete_rolling_history / len(featured):.2%}"
    )


if __name__ == "__main__":
    main()