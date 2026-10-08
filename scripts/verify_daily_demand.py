"""Verify DemandPilot daily SKU demand aggregation."""

from demandpilot.data.ingest import load_retail_data
from demandpilot.features.daily import build_daily_demand
from demandpilot.features.prepare import prepare_forecasting_transactions


def main() -> None:
    """Run daily aggregation against the complete retail dataset."""
    raw = load_retail_data()

    prepared, preparation_report = (
        prepare_forecasting_transactions(raw)
    )

    daily, daily_report = build_daily_demand(prepared)

    assert not daily.empty
    assert not daily.duplicated(["sku", "date"]).any()
    assert daily["sku"].notna().all()
    assert daily["date"].notna().all()
    assert (daily["demand"] > 0).all()

    assert (
        daily_report.total_units_input
        == daily_report.total_units_output
    )

    assert (
        daily_report.input_transaction_rows
        == preparation_report.output_rows
    )

    print("DemandPilot CP-02 daily demand aggregation PASSED")
    print(
        "Prepared transaction rows: "
        f"{daily_report.input_transaction_rows:,}"
    )
    print(
        "Observed SKU-day rows: "
        f"{daily_report.output_daily_rows:,}"
    )
    print(
        "Transaction rows collapsed: "
        f"{daily_report.rows_collapsed:,}"
    )
    print(
        f"Unique SKUs: {daily_report.unique_skus:,}"
    )
    print(
        f"Observed calendar days: {daily_report.calendar_days:,}"
    )
    print(
        f"Date range: {daily_report.date_min} "
        f"-> {daily_report.date_max}"
    )
    print(
        "Total units before aggregation: "
        f"{daily_report.total_units_input:,.0f}"
    )
    print(
        "Total units after aggregation: "
        f"{daily_report.total_units_output:,.0f}"
    )


if __name__ == "__main__":
    main()