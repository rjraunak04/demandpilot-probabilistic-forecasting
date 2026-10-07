"""Profiling utilities for DemandPilot retail transaction data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from demandpilot.data.ingest import PROJECT_ROOT, load_retail_data

DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "data_quality" / "data_quality_summary.json"
)


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-safe Python values."""
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


def profile_dataset(data: pd.DataFrame) -> dict[str, Any]:
    """Create a non-mutating audit summary of raw retail transactions."""
    required = {
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
    missing = required.difference(data.columns)

    if missing:
        raise ValueError(f"Cannot profile dataset; missing columns: {sorted(missing)}")

    invoice_date = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    invoice_text = data["Invoice"].astype("string")

    missing_counts = data.isna().sum()
    missing_rates = data.isna().mean()

    negative_quantity = data["Quantity"] < 0
    zero_quantity = data["Quantity"] == 0
    negative_price = data["Price"] < 0
    zero_price = data["Price"] == 0
    cancellation_invoice = invoice_text.str.startswith("C", na=False)

    duplicate_rows = data.duplicated()

    summary: dict[str, Any] = {
        "dataset": {
            "rows": int(len(data)),
            "columns": int(data.shape[1]),
            "column_names": data.columns.tolist(),
            "memory_mb": round(
                float(data.memory_usage(deep=True).sum() / (1024**2)),
                2,
            ),
        },
        "temporal": {
            "date_min": (
                invoice_date.min().isoformat()
                if invoice_date.notna().any()
                else None
            ),
            "date_max": (
                invoice_date.max().isoformat()
                if invoice_date.notna().any()
                else None
            ),
            "invalid_invoice_dates": int(invoice_date.isna().sum()),
            "unique_calendar_days": int(invoice_date.dt.date.nunique()),
        },
        "cardinality": {
            "unique_invoices": int(data["Invoice"].nunique(dropna=True)),
            "unique_skus": int(data["StockCode"].nunique(dropna=True)),
            "unique_customers": int(data["Customer ID"].nunique(dropna=True)),
            "unique_countries": int(data["Country"].nunique(dropna=True)),
        },
        "missingness": {
            column: {
                "count": int(missing_counts[column]),
                "rate": round(float(missing_rates[column]), 6),
            }
            for column in data.columns
        },
        "quantity": {
            "negative_count": int(negative_quantity.sum()),
            "zero_count": int(zero_quantity.sum()),
            "positive_count": int((data["Quantity"] > 0).sum()),
            "min": _json_safe(data["Quantity"].min()),
            "max": _json_safe(data["Quantity"].max()),
        },
        "price": {
            "negative_count": int(negative_price.sum()),
            "zero_count": int(zero_price.sum()),
            "positive_count": int((data["Price"] > 0).sum()),
            "min": _json_safe(data["Price"].min()),
            "max": _json_safe(data["Price"].max()),
        },
        "returns_and_cancellations": {
            "cancellation_invoice_count": int(cancellation_invoice.sum()),
            "negative_quantity_count": int(negative_quantity.sum()),
            "cancellation_rate": round(
                float(cancellation_invoice.mean()),
                6,
            ),
        },
        "duplicates": {
            "exact_duplicate_rows": int(duplicate_rows.sum()),
            "exact_duplicate_rate": round(float(duplicate_rows.mean()), 6),
        },
        "source_sheets": {
            str(sheet): int(count)
            for sheet, count in data["SourceSheet"].value_counts().items()
        },
        "top_countries": {
            str(country): int(count)
            for country, count in data["Country"].value_counts().head(10).items()
        },
        "top_skus": {
            str(sku): int(count)
            for sku, count in data["StockCode"].value_counts().head(10).items()
        },
    }

    return summary


def save_profile(
    summary: dict[str, Any],
    output_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Persist profiling results as a reproducible JSON artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    return output_path


def main() -> None:
    """Run the complete raw-data profiling workflow."""
    print("Loading Online Retail II...")
    data = load_retail_data()

    print("Profiling raw transactions...")
    summary = profile_dataset(data)
    output_path = save_profile(summary)

    print("DemandPilot data profiling PASSED")
    print(f"Rows: {summary['dataset']['rows']:,}")
    print(f"SKUs: {summary['cardinality']['unique_skus']:,}")
    print(f"Customers: {summary['cardinality']['unique_customers']:,}")
    print(f"Countries: {summary['cardinality']['unique_countries']:,}")
    print(
        "Negative quantities:",
        f"{summary['quantity']['negative_count']:,}",
    )
    print(
        "Cancellation invoices:",
        f"{summary['returns_and_cancellations']['cancellation_invoice_count']:,}",
    )
    print(
        "Exact duplicate rows:",
        f"{summary['duplicates']['exact_duplicate_rows']:,}",
    )
    print(f"Report: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()