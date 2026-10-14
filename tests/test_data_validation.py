"""Tests for DemandPilot raw retail data contracts."""

import pandas as pd
import pytest

from demandpilot.data.validate import (
    DataContractError,
    validate_retail_data,
)


def make_valid_data() -> pd.DataFrame:
    """Create a minimal valid retail transaction dataset."""
    return pd.DataFrame(
        {
            "Invoice": ["489434", "C489435", "489436"],
            "StockCode": ["85048", "79323P", "22041"],
            "Description": [
                "15CM CHRISTMAS GLASS BALL",
                "PINK CHERRY LIGHTS",
                None,
            ],
            "Quantity": [12, -4, 6],
            "InvoiceDate": [
                "2009-12-01 07:45:00",
                "2009-12-01 07:46:00",
                "2009-12-01 08:00:00",
            ],
            "Price": [6.95, 6.75, 0.0],
            "Customer ID": [13085.0, None, 13085.0],
            "Country": [
                "United Kingdom",
                "United Kingdom",
                "United Kingdom",
            ],
            "SourceSheet": [
                "Year 2009-2010",
                "Year 2009-2010",
                "Year 2009-2010",
            ],
        }
    )


def test_valid_dataset_passes_contracts() -> None:
    """Valid structural data should pass every enforced contract."""
    data = make_valid_data()

    report = validate_retail_data(data)

    assert report.passed
    assert report.rows == 3
    assert report.columns == 9
    assert all(report.checks.values())


def test_empty_dataset_is_rejected() -> None:
    """Empty input must fail before downstream processing."""
    data = make_valid_data().iloc[0:0]

    with pytest.raises(
        DataContractError,
        match="must not be empty",
    ):
        validate_retail_data(data)


def test_missing_required_column_is_rejected() -> None:
    """Removing a required source column must violate the contract."""
    data = make_valid_data().drop(columns=["StockCode"])

    with pytest.raises(
        DataContractError,
        match="Missing required columns",
    ):
        validate_retail_data(data)


@pytest.mark.parametrize(
    "column",
    ["Invoice", "StockCode", "InvoiceDate", "Country"],
)
def test_missing_core_identifier_is_rejected(column: str) -> None:
    """Core transaction identifiers cannot contain missing values."""
    data = make_valid_data()
    data.loc[0, column] = None

    with pytest.raises(
        DataContractError,
        match=column,
    ):
        validate_retail_data(data)


def test_non_numeric_quantity_is_rejected() -> None:
    """Quantity must remain numerically interpretable."""
    data = make_valid_data()

    # Explicitly use object dtype so malformed source-like data can be
    # constructed under strict pandas dtype rules.
    data["Quantity"] = data["Quantity"].astype("object")
    data.loc[0, "Quantity"] = "not-a-number"

    with pytest.raises(
        DataContractError,
        match="Quantity",
    ):
        validate_retail_data(data)


def test_non_numeric_price_is_rejected() -> None:
    """Price must remain numerically interpretable."""
    data = make_valid_data()

    # Explicitly use object dtype before injecting an invalid string.
    data["Price"] = data["Price"].astype("object")
    data.loc[0, "Price"] = "invalid-price"

    with pytest.raises(
        DataContractError,
        match="Price",
    ):
        validate_retail_data(data)


def test_invalid_invoice_date_is_rejected() -> None:
    """Malformed transaction timestamps must fail validation."""
    data = make_valid_data()
    data.loc[0, "InvoiceDate"] = "not-a-date"

    with pytest.raises(
        DataContractError,
        match="InvoiceDate",
    ):
        validate_retail_data(data)


def test_unexpected_source_sheet_is_rejected() -> None:
    """Unknown workbook provenance must violate the source contract."""
    data = make_valid_data()
    data.loc[0, "SourceSheet"] = "Unknown Sheet"

    with pytest.raises(
        DataContractError,
        match="Unexpected SourceSheet",
    ):
        validate_retail_data(data)


def test_missing_source_sheet_is_rejected() -> None:
    """Source provenance cannot be missing."""
    data = make_valid_data()
    data.loc[0, "SourceSheet"] = None

    with pytest.raises(
        DataContractError,
        match="SourceSheet contains missing values",
    ):
        validate_retail_data(data)


def test_negative_quantity_is_observed_not_rejected() -> None:
    """Returns should be recorded rather than rejected structurally."""
    data = make_valid_data()

    report = validate_retail_data(data)

    assert report.passed
    assert report.observations["negative_quantity_rows"] == 1
    assert report.observations["cancellation_invoice_rows"] == 1


def test_zero_price_is_observed_not_rejected() -> None:
    """Zero-price rows require business treatment, not structural failure."""
    data = make_valid_data()

    report = validate_retail_data(data)

    assert report.passed
    assert report.observations["zero_price_rows"] == 1


def test_missing_customer_is_observed_not_rejected() -> None:
    """Customer ID is optional for SKU-level demand construction."""
    data = make_valid_data()

    report = validate_retail_data(data)

    assert report.passed
    assert report.observations["missing_customer_rows"] == 1


def test_missing_description_is_observed_not_rejected() -> None:
    """Missing descriptions should not invalidate SKU transaction data."""
    data = make_valid_data()

    report = validate_retail_data(data)

    assert report.passed
    assert report.observations["missing_description_rows"] == 1