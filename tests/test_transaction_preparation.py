"""Tests for forecasting transaction preparation."""

import pandas as pd

from demandpilot.features.prepare import prepare_forecasting_transactions


def make_transactions() -> pd.DataFrame:
    """Create structurally valid transactions with business edge cases."""
    return pd.DataFrame(
        {
            "Invoice": [
                "10001",
                "10001",
                "C10002",
                "10003",
                "10004",
                "10005",
            ],
            "StockCode": [
                "SKU-A",
                "SKU-A",
                "SKU-B",
                "SKU-C",
                "SKU-D",
                "SKU-E",
            ],
            "Description": [
                "Product A",
                "Product A",
                "Product B",
                "Product C",
                "Product D",
                "Product E",
            ],
            "Quantity": [
                5,
                5,
                -2,
                -1,
                3,
                4,
            ],
            "InvoiceDate": [
                "2010-01-01 10:00:00",
                "2010-01-01 10:00:00",
                "2010-01-02 11:00:00",
                "2010-01-03 12:00:00",
                "2010-01-04 13:00:00",
                "2010-01-05 14:00:00",
            ],
            "Price": [
                10.0,
                10.0,
                8.0,
                4.0,
                0.0,
                -1.0,
            ],
            "Customer ID": [
                1.0,
                1.0,
                2.0,
                3.0,
                None,
                5.0,
            ],
            "Country": [
                "United Kingdom",
                "United Kingdom",
                "United Kingdom",
                "United Kingdom",
                "United Kingdom",
                "United Kingdom",
            ],
            "SourceSheet": [
                "Year 2009-2010",
                "Year 2009-2010",
                "Year 2009-2010",
                "Year 2009-2010",
                "Year 2009-2010",
                "Year 2009-2010",
            ],
        }
    )


def test_preparation_removes_exact_duplicates() -> None:
    data = make_transactions()

    prepared, report = prepare_forecasting_transactions(data)

    assert not prepared.duplicated().any()
    assert report.exact_duplicates_removed == 1


def test_preparation_removes_cancellations() -> None:
    data = make_transactions()

    prepared, report = prepare_forecasting_transactions(data)

    cancellation_mask = (
        prepared["Invoice"]
        .astype("string")
        .str.startswith("C", na=False)
    )

    assert not cancellation_mask.any()
    assert report.cancellation_rows_removed == 1


def test_preparation_removes_remaining_non_positive_quantity() -> None:
    data = make_transactions()

    prepared, report = prepare_forecasting_transactions(data)

    assert (prepared["Quantity"] > 0).all()
    assert report.non_positive_quantity_rows_removed == 1


def test_zero_and_negative_prices_are_not_demand_filters() -> None:
    data = make_transactions()

    prepared, _ = prepare_forecasting_transactions(data)

    assert (prepared["Price"] == 0).any()
    assert (prepared["Price"] < 0).any()


def test_missing_customer_id_is_allowed() -> None:
    data = make_transactions()

    prepared, _ = prepare_forecasting_transactions(data)

    assert prepared["Customer ID"].isna().any()


def test_preparation_is_chronologically_sorted() -> None:
    data = make_transactions().sample(
        frac=1.0,
        random_state=42,
    )

    prepared, _ = prepare_forecasting_transactions(data)

    assert prepared["InvoiceDate"].is_monotonic_increasing


def test_preparation_does_not_mutate_input() -> None:
    data = make_transactions()
    original = data.copy(deep=True)

    prepare_forecasting_transactions(data)

    pd.testing.assert_frame_equal(data, original)


def test_preparation_report_reconciles_rows() -> None:
    data = make_transactions()

    _, report = prepare_forecasting_transactions(data)

    assert report.input_rows - report.rows_removed == report.output_rows
    assert report.rows_removed == (
        report.exact_duplicates_removed
        + report.cancellation_rows_removed
        + report.non_positive_quantity_rows_removed
        + report.missing_stock_code_rows_removed
        + report.missing_invoice_date_rows_removed
    )