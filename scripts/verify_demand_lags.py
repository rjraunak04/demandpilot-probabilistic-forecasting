"""Verify leakage-safe lag features on the complete retail dataset."""

from demandpilot.data.ingest import load_retail_data
from demandpilot.features.daily import build_daily_demand
from demandpilot.features.lags import (
    DEFAULT_LAGS,
    add_demand_lags,
)
from demandpilot.features.panel import reconstruct_zero_demand_days
from demandpilot.features.prepare import prepare_forecasting_transactions


def main() -> None:
    """Run DemandPilot pipeline through lag-feature generation."""
    raw = load_retail_data()

    prepared, _ = prepare_forecasting_transactions(raw)
    daily, _ = build_daily_demand(prepared)
    panel, panel_report = reconstruct_zero_demand_days(daily)
    featured, lag_report = add_demand_lags(panel)

    assert len(featured) == len(panel)
    assert not featured.duplicated(["sku", "date"]).any()

    assert (
        featured["demand"].sum()
        == panel["demand"].sum()
    )

    for lag in DEFAULT_LAGS:
        column = f"lag_{lag}"
        assert column in featured.columns
        assert (featured[column].dropna() >= 0).all()

    for _, sku_data in featured.groupby(
        "sku",
        observed=True,
    ):
        sku_data = sku_data.reset_index(drop=True)

        for lag in DEFAULT_LAGS:
            column = f"lag_{lag}"

            if len(sku_data) > lag:
                expected = sku_data["demand"].shift(lag)

                assert sku_data[column].equals(expected)

    lag_columns = [
        f"lag_{lag}"
        for lag in DEFAULT_LAGS
    ]

    complete_history = (
        featured[lag_columns]
        .notna()
        .all(axis=1)
    )

    assert (
        int(complete_history.sum())
        == lag_report.rows_with_complete_lag_history
    )

    assert (
        lag_report.rows_with_complete_lag_history
        + lag_report.rows_without_complete_lag_history
        == len(featured)
    )

    assert (
        lag_report.output_rows
        == panel_report.panel_rows
    )

    print("DemandPilot CP-04 leakage-safe lag features PASSED")
    print(
        f"Input panel rows: "
        f"{lag_report.input_rows:,}"
    )
    print(
        f"Output feature rows: "
        f"{lag_report.output_rows:,}"
    )
    print(
        f"Unique SKUs: "
        f"{lag_report.unique_skus:,}"
    )
    print(
        "Lag features: "
        + ", ".join(
            f"lag_{lag}"
            for lag in lag_report.lag_days
        )
    )
    print(
        "Rows with complete lag history: "
        f"{lag_report.rows_with_complete_lag_history:,}"
    )
    print(
        "Rows without complete lag history: "
        f"{lag_report.rows_without_complete_lag_history:,}"
    )
    print(
        "Complete-history share: "
        f"{lag_report.rows_with_complete_lag_history / len(featured):.2%}"
    )


if __name__ == "__main__":
    main()