"""Construct continuous SKU-level daily demand panels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PanelReport:
    """Audit information for zero-demand calendar reconstruction."""

    observed_rows: int
    panel_rows: int
    inserted_zero_rows: int
    unique_skus: int
    date_min: str | None
    date_max: str | None
    total_demand_before: float
    total_demand_after: float

    @property
    def expansion_rows(self) -> int:
        """Return the number of rows added during calendar reconstruction."""
        return self.panel_rows - self.observed_rows

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "observed_rows": self.observed_rows,
            "panel_rows": self.panel_rows,
            "inserted_zero_rows": self.inserted_zero_rows,
            "expansion_rows": self.expansion_rows,
            "unique_skus": self.unique_skus,
            "date_min": self.date_min,
            "date_max": self.date_max,
            "total_demand_before": self.total_demand_before,
            "total_demand_after": self.total_demand_after,
        }


def _validate_daily_demand(daily: pd.DataFrame) -> None:
    """Validate assumptions required for panel reconstruction."""
    required_columns = {
        "sku",
        "date",
        "demand",
    }
    missing = required_columns.difference(daily.columns)

    if missing:
        raise ValueError(
            f"Missing columns required for panel reconstruction: "
            f"{sorted(missing)}"
        )

    if daily.empty:
        raise ValueError(
            "Observed daily demand dataset must not be empty."
        )

    if daily["sku"].isna().any():
        raise ValueError(
            "SKU must not contain missing values."
        )

    if daily["date"].isna().any():
        raise ValueError(
            "Date must not contain missing values."
        )

    if daily["demand"].isna().any():
        raise ValueError(
            "Demand must not contain missing values."
        )

    if daily.duplicated(["sku", "date"]).any():
        raise ValueError(
            "Observed daily demand must have unique SKU-day grain."
        )

    demand = pd.to_numeric(
        daily["demand"],
        errors="coerce",
    )

    if demand.isna().any():
        raise ValueError(
            "Demand must contain numeric values."
        )

    if (demand <= 0).any():
        raise ValueError(
            "Observed daily demand must be strictly positive."
        )


def reconstruct_zero_demand_days(
    daily: pd.DataFrame,
) -> tuple[pd.DataFrame, PanelReport]:
    """Build a continuous daily calendar within each SKU's observed lifetime.

    Each SKU is expanded from its first observed positive-demand date through
    its last observed positive-demand date. Missing dates inside that interval
    are represented as zero demand.

    Dates before the first observed demand or after the last observed demand
    are not created.
    """
    _validate_daily_demand(daily)

    data = daily.copy()

    data["sku"] = (
        data["sku"]
        .astype("string")
        .str.strip()
    )
    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
        format="mixed",
    ).dt.normalize()
    data["demand"] = pd.to_numeric(
        data["demand"],
        errors="raise",
    )

    if data.duplicated(["sku", "date"]).any():
        raise ValueError(
            "Date normalization produced duplicate SKU-day rows."
        )

    data = data.sort_values(
        ["sku", "date"],
        kind="stable",
    ).reset_index(drop=True)

    observed_rows = int(len(data))
    total_demand_before = float(data["demand"].sum())

    panel_parts: list[pd.DataFrame] = []

    for sku, sku_data in data.groupby(
        "sku",
        sort=False,
        observed=True,
    ):
        sku_data = sku_data.sort_values(
            "date",
            kind="stable",
        )

        start_date = sku_data["date"].iloc[0]
        end_date = sku_data["date"].iloc[-1]

        calendar = pd.DataFrame(
            {
                "date": pd.date_range(
                    start=start_date,
                    end=end_date,
                    freq="D",
                )
            }
        )

        calendar["sku"] = sku

        sku_panel = calendar.merge(
            sku_data[["sku", "date", "demand"]],
            on=["sku", "date"],
            how="left",
            validate="one_to_one",
        )

        sku_panel["is_observed_demand"] = (
            sku_panel["demand"].notna()
        )

        sku_panel["demand"] = (
            sku_panel["demand"]
            .fillna(0)
        )

        panel_parts.append(sku_panel)

    panel = pd.concat(
        panel_parts,
        ignore_index=True,
    )

    panel = panel[
        [
            "sku",
            "date",
            "demand",
            "is_observed_demand",
        ]
    ]

    panel = panel.sort_values(
        ["sku", "date"],
        kind="stable",
    ).reset_index(drop=True)

    if panel.duplicated(["sku", "date"]).any():
        raise RuntimeError(
            "Panel reconstruction produced duplicate SKU-day rows."
        )

    if (panel["demand"] < 0).any():
        raise RuntimeError(
            "Panel reconstruction produced negative demand."
        )

    total_demand_after = float(panel["demand"].sum())

    if total_demand_before != total_demand_after:
        raise RuntimeError(
            "Total demand changed during zero-demand reconstruction."
        )

    inserted_zero_rows = int(
        (~panel["is_observed_demand"]).sum()
    )

    if inserted_zero_rows != len(panel) - observed_rows:
        raise RuntimeError(
            "Inserted zero-demand row count does not reconcile."
        )

    date_min = panel["date"].min()
    date_max = panel["date"].max()

    report = PanelReport(
        observed_rows=observed_rows,
        panel_rows=int(len(panel)),
        inserted_zero_rows=inserted_zero_rows,
        unique_skus=int(panel["sku"].nunique()),
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
        total_demand_before=total_demand_before,
        total_demand_after=total_demand_after,
    )

    return panel, report