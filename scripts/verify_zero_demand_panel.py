"""Verify zero-demand SKU calendar reconstruction on the retail dataset."""

from demandpilot.data.ingest import load_retail_data
from demandpilot.features.daily import build_daily_demand
from demandpilot.features.panel import reconstruct_zero_demand_days
from demandpilot.features.prepare import prepare_forecasting_transactions


def main() -> None:
    """Run the complete pipeline through zero-demand reconstruction."""
    raw = load_retail_data()

    prepared, _ = prepare_forecasting_transactions(raw)
    daily, daily_report = build_daily_demand(prepared)
    panel, panel_report = reconstruct_zero_demand_days(daily)

    assert not panel.empty
    assert not panel.duplicated(["sku", "date"]).any()
    assert panel["sku"].notna().all()
    assert panel["date"].notna().all()
    assert panel["demand"].notna().all()
    assert (panel["demand"] >= 0).all()

    assert panel_report.observed_rows == daily_report.output_daily_rows

    assert (
        panel_report.total_demand_before
        == panel_report.total_demand_after
    )

    assert (
        panel_report.inserted_zero_rows
        == panel_report.panel_rows - panel_report.observed_rows
    )

    zero_rows = int((panel["demand"] == 0).sum())

    assert zero_rows == panel_report.inserted_zero_rows

    print("DemandPilot CP-03 zero-demand reconstruction PASSED")
    print(
        f"Observed SKU-day rows: "
        f"{panel_report.observed_rows:,}"
    )
    print(
        f"Continuous panel rows: "
        f"{panel_report.panel_rows:,}"
    )
    print(
        f"Inserted zero-demand rows: "
        f"{panel_report.inserted_zero_rows:,}"
    )
    print(
        "Panel expansion: "
        f"{panel_report.expansion_rows:,}"
    )
    print(
        f"Unique SKUs: "
        f"{panel_report.unique_skus:,}"
    )
    print(
        f"Date range: "
        f"{panel_report.date_min} -> "
        f"{panel_report.date_max}"
    )
    print(
        "Total demand before reconstruction: "
        f"{panel_report.total_demand_before:,.0f}"
    )
    print(
        "Total demand after reconstruction: "
        f"{panel_report.total_demand_after:,.0f}"
    )
    print(
        "Zero-demand share: "
        f"{panel_report.inserted_zero_rows / panel_report.panel_rows:.2%}"
    )


if __name__ == "__main__":
    main()