"""Run the DemandPilot retail ingestion smoke test."""

from demandpilot.data.ingest import load_retail_data, summarize_dataset


def main() -> None:
    """Verify that the configured retail workbook can be ingested."""
    data = load_retail_data()
    summary = summarize_dataset(data)

    assert not data.empty, "Retail dataset must not be empty."
    assert summary["rows"] > 0, "Retail dataset must contain rows."
    assert len(summary["source_sheets"]) == 2, "Expected two source worksheets."

    required_columns = {
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country",
        "SourceSheet",
    }

    missing_columns = required_columns.difference(data.columns)

    assert not missing_columns, (
        f"Missing expected ingestion columns: {sorted(missing_columns)}"
    )

    print("DemandPilot ingestion verification PASSED")
    print(f"Rows: {summary['rows']:,}")
    print(f"Columns: {summary['columns']}")
    print(f"Date range: {summary['date_min']} -> {summary['date_max']}")
    print(f"Source sheets: {summary['source_sheets']}")


if __name__ == "__main__":
    main()