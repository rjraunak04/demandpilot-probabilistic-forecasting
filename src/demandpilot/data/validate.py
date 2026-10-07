"""Data-quality contracts for DemandPilot raw retail transactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from demandpilot.data.ingest import load_retail_data

REQUIRED_COLUMNS = {
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

EXPECTED_SOURCE_SHEETS = {
    "Year 2009-2010",
    "Year 2010-2011",
}


class DataContractError(ValueError):
    """Raised when retail data violates a required structural contract."""


@dataclass
class ValidationReport:
    """Machine-readable result of raw retail data validation."""

    rows: int
    columns: int
    checks: dict[str, bool] = field(default_factory=dict)
    observations: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Return True when every enforced contract passed."""
        return all(self.checks.values())


def _require_columns(data: pd.DataFrame) -> None:
    """Require all columns expected by the raw retail contract."""
    missing = REQUIRED_COLUMNS.difference(data.columns)

    if missing:
        raise DataContractError(
            f"Missing required columns: {sorted(missing)}"
        )


def _require_non_empty(data: pd.DataFrame) -> None:
    """Reject empty input datasets."""
    if data.empty:
        raise DataContractError("Retail dataset must not be empty.")


def _require_core_identifiers(data: pd.DataFrame) -> None:
    """Require identifiers needed to interpret transaction rows."""
    for column in ("Invoice", "StockCode", "InvoiceDate", "Country"):
        missing_count = int(data[column].isna().sum())

        if missing_count:
            raise DataContractError(
                f"{column} contains {missing_count:,} missing values."
            )


def _require_numeric_quantity(data: pd.DataFrame) -> None:
    """Require Quantity to contain numeric values."""
    converted = pd.to_numeric(data["Quantity"], errors="coerce")
    invalid = converted.isna() & data["Quantity"].notna()

    if invalid.any():
        raise DataContractError(
            f"Quantity contains {int(invalid.sum()):,} non-numeric values."
        )


def _require_numeric_price(data: pd.DataFrame) -> None:
    """Require Price to contain numeric values."""
    converted = pd.to_numeric(data["Price"], errors="coerce")
    invalid = converted.isna() & data["Price"].notna()

    if invalid.any():
        raise DataContractError(
            f"Price contains {int(invalid.sum()):,} non-numeric values."
        )


def _require_parseable_dates(data: pd.DataFrame) -> None:
    """Require every non-null InvoiceDate value to be parseable."""
    parsed = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    invalid = parsed.isna() & data["InvoiceDate"].notna()

    if invalid.any():
        raise DataContractError(
            f"InvoiceDate contains {int(invalid.sum()):,} invalid values."
        )


def _require_source_provenance(data: pd.DataFrame) -> None:
    """Require source-sheet provenance to match configured workbook sheets."""
    if data["SourceSheet"].isna().any():
        raise DataContractError("SourceSheet contains missing values.")

    observed = set(data["SourceSheet"].astype(str).unique())
    unexpected = observed.difference(EXPECTED_SOURCE_SHEETS)

    if unexpected:
        raise DataContractError(
            f"Unexpected SourceSheet values: {sorted(unexpected)}"
        )


def validate_retail_data(data: pd.DataFrame) -> ValidationReport:
    """Validate raw retail data without cleaning or modifying it."""
    _require_non_empty(data)
    _require_columns(data)
    _require_core_identifiers(data)
    _require_numeric_quantity(data)
    _require_numeric_price(data)
    _require_parseable_dates(data)
    _require_source_provenance(data)

    quantity = pd.to_numeric(data["Quantity"], errors="coerce")
    price = pd.to_numeric(data["Price"], errors="coerce")
    invoice = data["Invoice"].astype("string")

    report = ValidationReport(
        rows=int(len(data)),
        columns=int(data.shape[1]),
        checks={
            "non_empty": True,
            "required_columns": True,
            "core_identifiers_present": True,
            "quantity_numeric": True,
            "price_numeric": True,
            "invoice_dates_parseable": True,
            "source_provenance_valid": True,
        },
        observations={
            "negative_quantity_rows": int((quantity < 0).sum()),
            "zero_quantity_rows": int((quantity == 0).sum()),
            "negative_price_rows": int((price < 0).sum()),
            "zero_price_rows": int((price == 0).sum()),
            "cancellation_invoice_rows": int(
                invoice.str.startswith("C", na=False).sum()
            ),
            "exact_duplicate_rows": int(data.duplicated().sum()),
            "missing_description_rows": int(
                data["Description"].isna().sum()
            ),
            "missing_customer_rows": int(
                data["Customer ID"].isna().sum()
            ),
        },
    )

    return report


def main() -> None:
    """Validate the configured real retail dataset."""
    data = load_retail_data()
    report = validate_retail_data(data)

    print("DemandPilot retail data contract PASSED")
    print(f"Rows validated: {report.rows:,}")
    print(f"Columns: {report.columns}")
    print(f"Contract checks: {len(report.checks)}")
    print(f"Passed: {report.passed}")

    for name, value in report.observations.items():
        print(f"{name}: {value:,}")


if __name__ == "__main__":
    main()