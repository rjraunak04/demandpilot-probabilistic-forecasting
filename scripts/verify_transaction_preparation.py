"""Verify DemandPilot forecasting transaction preparation."""

from demandpilot.data.ingest import load_retail_data
from demandpilot.features.prepare import prepare_forecasting_transactions


def main() -> None:
    """Run transaction preparation against the complete retail dataset."""
    raw = load_retail_data()

    prepared, report = prepare_forecasting_transactions(raw)

    assert not prepared.empty
    assert not prepared.duplicated().any()
    assert prepared["StockCode"].notna().all()
    assert prepared["InvoiceDate"].notna().all()
    assert (prepared["Quantity"] > 0).all()

    cancellation_rows = (
        prepared["Invoice"]
        .astype("string")
        .str.startswith("C", na=False)
        .sum()
    )
    assert cancellation_rows == 0

    print("DemandPilot CP-01 transaction preparation PASSED")
    print(f"Input rows: {report.input_rows:,}")
    print(f"Output rows: {report.output_rows:,}")
    print(f"Rows removed: {report.rows_removed:,}")
    print(
        "Exact duplicates removed: "
        f"{report.exact_duplicates_removed:,}"
    )
    print(
        "Cancellation rows removed: "
        f"{report.cancellation_rows_removed:,}"
    )
    print(
        "Remaining non-positive quantity rows removed: "
        f"{report.non_positive_quantity_rows_removed:,}"
    )
    print(
        "Missing StockCode rows removed: "
        f"{report.missing_stock_code_rows_removed:,}"
    )
    print(
        "Missing InvoiceDate rows removed: "
        f"{report.missing_invoice_date_rows_removed:,}"
    )


if __name__ == "__main__":
    main()