"""Build daily SKU-level demand observations from prepared transactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DailyDemandReport:
    """Audit information for daily SKU demand aggregation."""

    input_transaction_rows: int
    output_daily_rows: int
    unique_skus: int
    calendar_days: int
    date_min: str | None
    date_max: str | None
    total_units_input: float
    total_units_output: float

    @property
    def rows_collapsed(self) -> int:
        """Return transaction rows collapsed by daily aggregation."""
        return self.input_transaction_rows - self.output_daily_rows

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "input_transaction_rows": self.input_transaction_rows,
            "output_daily_rows": self.output_daily_rows,
            "rows_collapsed": self.rows_collapsed,
            "unique_skus": self.unique_skus,
            "calendar_days": self.calendar_days,
            "date_min": self.date_min,
            "date_max": self.date_max,
            "total_units_input": self.total_units_input,
            "total_units_output": self.total_units_output,
        }


def build_daily_demand(
    transactions: pd.DataFrame,
) -> tuple[pd.DataFrame, DailyDemandReport]:
    """Aggregate prepared positive-demand transactions to SKU-day grain.

    Parameters
    ----------
    transactions:
        Forecasting-ready transaction data containing at least StockCode,
        InvoiceDate, and Quantity.

    Returns
    -------
    tuple[pd.DataFrame, DailyDemandReport]
        Unique observed SKU-day demand rows and aggregation audit metadata.

    Notes
    -----
    This function only aggregates observed positive-demand transactions.
    Missing SKU-day combinations are not interpreted as zero demand here.
    Continuous calendar reconstruction belongs to the next pipeline stage.
    """
    required_columns = {
        "StockCode",
        "InvoiceDate",
        "Quantity",
    }
    missing = required_columns.difference(transactions.columns)

    if missing:
        raise ValueError(
            f"Missing columns required for daily demand: {sorted(missing)}"
        )

    if transactions.empty:
        raise ValueError(
            "Prepared transaction dataset must not be empty."
        )

    data = transactions.copy()

    data["InvoiceDate"] = pd.to_datetime(
        data["InvoiceDate"],
        errors="raise",
        format="mixed",
    )
    data["Quantity"] = pd.to_numeric(
        data["Quantity"],
        errors="raise",
    )
    data["StockCode"] = (
        data["StockCode"]
        .astype("string")
        .str.strip()
    )

    if data["StockCode"].isna().any():
        raise ValueError(
            "StockCode must not contain missing values."
        )

    if data["InvoiceDate"].isna().any():
        raise ValueError(
            "InvoiceDate must not contain missing values."
        )

    if (data["Quantity"] <= 0).any():
        raise ValueError(
            "Daily demand aggregation requires positive quantities."
        )

    input_rows = int(len(data))
    total_units_input = float(data["Quantity"].sum())

    data["date"] = data["InvoiceDate"].dt.normalize()

    daily = (
        data.groupby(
            ["StockCode", "date"],
            as_index=False,
            observed=True,
        )["Quantity"]
        .sum()
        .rename(
            columns={
                "StockCode": "sku",
                "Quantity": "demand",
            }
        )
    )

    daily = daily.sort_values(
        ["sku", "date"],
        kind="stable",
    ).reset_index(drop=True)

    if daily.duplicated(["sku", "date"]).any():
        raise RuntimeError(
            "Daily demand aggregation produced duplicate SKU-day rows."
        )

    if (daily["demand"] <= 0).any():
        raise RuntimeError(
            "Daily demand aggregation produced non-positive demand."
        )

    total_units_output = float(daily["demand"].sum())

    if total_units_input != total_units_output:
        raise RuntimeError(
            "Demand units were not preserved during daily aggregation."
        )

    date_min = daily["date"].min()
    date_max = daily["date"].max()

    report = DailyDemandReport(
        input_transaction_rows=input_rows,
        output_daily_rows=int(len(daily)),
        unique_skus=int(daily["sku"].nunique()),
        calendar_days=int(daily["date"].nunique()),
        date_min=(
            date_min.isoformat()
            if pd.notna(date_min)
            else None
        ),
        date_max=(
            date_max.isoformat()
            if pd.notna(date_max)
            else None
        ),
        total_units_input=total_units_input,
        total_units_output=total_units_output,
    )

    return daily, report