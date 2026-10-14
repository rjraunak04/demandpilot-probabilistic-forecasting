"""Tests for daily SKU demand aggregation."""

import pandas as pd
import pytest

from demandpilot.features.daily import build_daily_demand


def make_transactions() -> pd.DataFrame:
    """Create positive prepared transactions for aggregation tests."""
    return pd.DataFrame(
        {
            "StockCode": [
                "SKU-A",
                "SKU-A",
                "SKU-A",
                "SKU-B",
                "SKU-B",
            ],
            "InvoiceDate": [
                "2010-01-01 09:00:00",
                "2010-01-01 15:30:00",
                "2010-01-03 10:00:00",
                "2010-01-01 11:00:00",
                "2010-01-02 12:00:00",
            ],
            "Quantity": [
                2,
                3,
                4,
                10,
                5,
            ],
        }
    )


def test_daily_demand_aggregates_same_sku_day() -> None:
    transactions = make_transactions()

    daily, _ = build_daily_demand(transactions)

    row = daily.loc[
        (daily["sku"] == "SKU-A")
        & (daily["date"] == pd.Timestamp("2010-01-01"))
    ]

    assert len(row) == 1
    assert row.iloc[0]["demand"] == 5


def test_daily_demand_has_unique_grain() -> None:
    transactions = make_transactions()

    daily, _ = build_daily_demand(transactions)

    assert not daily.duplicated(["sku", "date"]).any()


def test_daily_demand_preserves_total_units() -> None:
    transactions = make_transactions()

    daily, report = build_daily_demand(transactions)

    assert daily["demand"].sum() == transactions["Quantity"].sum()
    assert report.total_units_input == report.total_units_output


def test_daily_demand_does_not_fill_missing_days() -> None:
    transactions = make_transactions()

    daily, _ = build_daily_demand(transactions)

    sku_a = daily.loc[daily["sku"] == "SKU-A"]

    assert len(sku_a) == 2
    assert pd.Timestamp("2010-01-02") not in set(sku_a["date"])


def test_daily_demand_keeps_skus_separate() -> None:
    transactions = make_transactions()

    daily, _ = build_daily_demand(transactions)

    sku_a_day_one = daily.loc[
        (daily["sku"] == "SKU-A")
        & (daily["date"] == pd.Timestamp("2010-01-01")),
        "demand",
    ].iloc[0]

    sku_b_day_one = daily.loc[
        (daily["sku"] == "SKU-B")
        & (daily["date"] == pd.Timestamp("2010-01-01")),
        "demand",
    ].iloc[0]

    assert sku_a_day_one == 5
    assert sku_b_day_one == 10


def test_daily_demand_is_sorted_by_sku_and_date() -> None:
    transactions = make_transactions().sample(
        frac=1.0,
        random_state=42,
    )

    daily, _ = build_daily_demand(transactions)

    expected = daily.sort_values(
        ["sku", "date"],
        kind="stable",
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(daily, expected)


def test_daily_demand_rejects_empty_input() -> None:
    transactions = make_transactions().iloc[0:0]

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        build_daily_demand(transactions)


def test_daily_demand_rejects_missing_required_column() -> None:
    transactions = make_transactions().drop(
        columns=["Quantity"]
    )

    with pytest.raises(
        ValueError,
        match="Missing columns required",
    ):
        build_daily_demand(transactions)


def test_daily_demand_rejects_non_positive_quantity() -> None:
    transactions = make_transactions()
    transactions.loc[0, "Quantity"] = 0

    with pytest.raises(
        ValueError,
        match="requires positive quantities",
    ):
        build_daily_demand(transactions)


def test_daily_demand_report_reconciles_rows() -> None:
    transactions = make_transactions()

    daily, report = build_daily_demand(transactions)

    assert report.input_transaction_rows == len(transactions)
    assert report.output_daily_rows == len(daily)
    assert (
        report.rows_collapsed
        == report.input_transaction_rows
        - report.output_daily_rows
    )