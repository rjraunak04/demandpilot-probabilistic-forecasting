"""Run and verify the DemandPilot raw retail data audit."""

from demandpilot.data.ingest import load_retail_data
from demandpilot.data.profile import profile_dataset, save_profile


def main() -> None:
    """Profile real retail data and enforce audit-level sanity checks."""
    data = load_retail_data()
    summary = profile_dataset(data)

    assert summary["dataset"]["rows"] > 0
    assert summary["cardinality"]["unique_skus"] > 0
    assert summary["cardinality"]["unique_countries"] > 0
    assert summary["temporal"]["date_min"] is not None
    assert summary["temporal"]["date_max"] is not None

    output_path = save_profile(summary)

    print("DemandPilot CP-03 data audit PASSED")
    print(f"Transactions: {summary['dataset']['rows']:,}")
    print(f"Unique invoices: {summary['cardinality']['unique_invoices']:,}")
    print(f"Unique SKUs: {summary['cardinality']['unique_skus']:,}")
    print(f"Unique customers: {summary['cardinality']['unique_customers']:,}")
    print(f"Countries: {summary['cardinality']['unique_countries']:,}")
    print(f"Calendar days: {summary['temporal']['unique_calendar_days']:,}")
    print(f"Negative quantities: {summary['quantity']['negative_count']:,}")
    print(f"Zero quantities: {summary['quantity']['zero_count']:,}")
    print(f"Zero prices: {summary['price']['zero_count']:,}")
    print(
        "Cancellation invoices:",
        f"{summary['returns_and_cancellations']['cancellation_invoice_count']:,}",
    )
    print(
        "Exact duplicate rows:",
        f"{summary['duplicates']['exact_duplicate_rows']:,}",
    )
    print(f"Audit artifact: {output_path}")


if __name__ == "__main__":
    main()