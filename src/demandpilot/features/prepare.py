"""Prepare validated retail transactions for demand forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from demandpilot.data.validate import validate_retail_data


@dataclass(frozen=True)
class PreparationReport:
    """Audit information for forecasting transaction preparation."""

    input_rows: int
    output_rows: int
    exact_duplicates_removed: int
    cancellation_rows_removed: int
    non_positive_quantity_rows_removed: int
    missing_stock_code_rows_removed: int
    missing_invoice_date_rows_removed: int

    @property
    def rows_removed(self) -> int:
        """Return the total number of removed transaction rows."""
        return self.input_rows - self.output_rows

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the report."""
        return {
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "rows_removed": self.rows_removed,
            "exact_duplicates_removed": self.exact_duplicates_removed,
            "cancellation_rows_removed": self.cancellation_rows_removed,
            "non_positive_quantity_rows_removed": (
                self.non_positive_quantity_rows_removed
            ),
            "missing_stock_code_rows_removed": (
                self.missing_stock_code_rows_removed
            ),
            "missing_invoice_date_rows_removed": (
                self.missing_invoice_date_rows_removed
            ),
        }


def prepare_forecasting_transactions(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, PreparationReport]:
    """Create positive realized-demand transactions from validated raw data.

    The function intentionally does not aggregate transactions to daily demand.
    That transformation belongs to the next forecasting pipeline stage.
    """
    validate_retail_data(data)

    prepared = data.copy()
    input_rows = len(prepared)

    duplicate_mask = prepared.duplicated()
    exact_duplicates_removed = int(duplicate_mask.sum())
    prepared = prepared.loc[~duplicate_mask].copy()

    invoice = prepared["Invoice"].astype("string").str.strip()
    cancellation_mask = invoice.str.startswith("C", na=False)
    cancellation_rows_removed = int(cancellation_mask.sum())
    prepared = prepared.loc[~cancellation_mask].copy()

    quantity = pd.to_numeric(
        prepared["Quantity"],
        errors="coerce",
    )
    non_positive_quantity_mask = quantity <= 0
    non_positive_quantity_rows_removed = int(
        non_positive_quantity_mask.sum()
    )
    prepared = prepared.loc[~non_positive_quantity_mask].copy()

    missing_stock_code_mask = prepared["StockCode"].isna()
    missing_stock_code_rows_removed = int(
        missing_stock_code_mask.sum()
    )
    prepared = prepared.loc[~missing_stock_code_mask].copy()

    missing_invoice_date_mask = prepared["InvoiceDate"].isna()
    missing_invoice_date_rows_removed = int(
        missing_invoice_date_mask.sum()
    )
    prepared = prepared.loc[~missing_invoice_date_mask].copy()

    prepared["InvoiceDate"] = pd.to_datetime(
        prepared["InvoiceDate"],
        errors="raise",
        format="mixed",
    )

    prepared["StockCode"] = (
        prepared["StockCode"]
        .astype("string")
        .str.strip()
    )

    prepared["Quantity"] = pd.to_numeric(
        prepared["Quantity"],
        errors="raise",
    )

    prepared = prepared.sort_values(
        ["InvoiceDate", "StockCode", "Invoice"],
        kind="stable",
    ).reset_index(drop=True)

    report = PreparationReport(
        input_rows=int(input_rows),
        output_rows=int(len(prepared)),
        exact_duplicates_removed=exact_duplicates_removed,
        cancellation_rows_removed=cancellation_rows_removed,
        non_positive_quantity_rows_removed=(
            non_positive_quantity_rows_removed
        ),
        missing_stock_code_rows_removed=missing_stock_code_rows_removed,
        missing_invoice_date_rows_removed=missing_invoice_date_rows_removed,
    )

    return prepared, report