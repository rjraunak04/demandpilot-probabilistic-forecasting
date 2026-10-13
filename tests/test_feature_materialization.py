"""Tests for forecasting feature artifact materialization."""

from datetime import timedelta

import json
from pathlib import Path

import pandas as pd

from demandpilot.features.materialize import (
    materialize_feature_pipeline,
)
from demandpilot.features.validate import (
    validate_forecasting_features,
)


def make_raw_transactions() -> pd.DataFrame:
    """Create a small deterministic transaction dataset."""
    rows: list[dict[str, object]] = []

    dates = pd.date_range(
        "2010-01-01",
        periods=35,
        freq="D",
    )

    for sku_index, sku in enumerate(
        ("SKU-A", "SKU-B")
    ):
        for day_index, date in enumerate(dates):
            if day_index in {4, 11, 19}:
                continue

            rows.append(
                {
                    "Invoice": (
                        f"{sku_index + 1}"
                        f"{day_index:05d}"
                    ),
                    "StockCode": sku,
                    "Description": f"{sku} product",
                    "Quantity": day_index + sku_index + 1,
                    "InvoiceDate": (
                        date + timedelta(hours=10)
                    ),
                    "Price": 2.5,
                    "Customer ID": 1000 + sku_index,
                    "Country": "United Kingdom",
                    "SourceSheet": "Year 2009-2010",
                }
            )

    return pd.DataFrame(rows)


def test_materialization_creates_expected_artifacts(
    tmp_path: Path,
) -> None:
    """Materialization should create both Parquet files and JSON summary."""
    raw = make_raw_transactions()

    daily_path = tmp_path / "daily.parquet"
    feature_path = tmp_path / "features.parquet"
    summary_path = tmp_path / "summary.json"

    materialize_feature_pipeline(
        raw,
        daily_path=daily_path,
        feature_path=feature_path,
        summary_path=summary_path,
    )

    assert daily_path.exists()
    assert feature_path.exists()
    assert summary_path.exists()


def test_materialized_features_satisfy_contract(
    tmp_path: Path,
) -> None:
    """Persisted feature data should remain contract-valid."""
    raw = make_raw_transactions()

    feature_path = tmp_path / "features.parquet"

    materialize_feature_pipeline(
        raw,
        daily_path=tmp_path / "daily.parquet",
        feature_path=feature_path,
        summary_path=tmp_path / "summary.json",
    )

    features = pd.read_parquet(
        feature_path
    )

    validate_forecasting_features(
        features
    )


def test_materialized_daily_dataset_has_expected_grain(
    tmp_path: Path,
) -> None:
    """Daily artifact should contain unique observed SKU-day rows."""
    raw = make_raw_transactions()

    daily_path = tmp_path / "daily.parquet"

    materialize_feature_pipeline(
        raw,
        daily_path=daily_path,
        feature_path=tmp_path / "features.parquet",
        summary_path=tmp_path / "summary.json",
    )

    daily = pd.read_parquet(
        daily_path
    )

    assert list(daily.columns) == [
        "sku",
        "date",
        "demand",
    ]

    assert not daily.duplicated(
        ["sku", "date"]
    ).any()

    assert (daily["demand"] > 0).all()


def test_summary_reconciles_with_materialized_features(
    tmp_path: Path,
) -> None:
    """Summary counts should match the persisted feature dataset."""
    raw = make_raw_transactions()

    feature_path = tmp_path / "features.parquet"
    summary_path = tmp_path / "summary.json"

    materialize_feature_pipeline(
        raw,
        daily_path=tmp_path / "daily.parquet",
        feature_path=feature_path,
        summary_path=summary_path,
    )

    features = pd.read_parquet(
        feature_path
    )

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        summary["feature_dataset"]["rows"]
        == len(features)
    )

    assert (
        summary["feature_dataset"]["columns"]
        == len(features.columns)
    )

    assert (
        summary["feature_dataset"]["unique_skus"]
        == features["sku"].nunique()
    )


def test_summary_records_artifact_locations(
    tmp_path: Path,
) -> None:
    """Summary should identify the generated artifact locations."""
    raw = make_raw_transactions()

    daily_path = tmp_path / "daily.parquet"
    feature_path = tmp_path / "features.parquet"
    summary_path = tmp_path / "summary.json"

    summary = materialize_feature_pipeline(
        raw,
        daily_path=daily_path,
        feature_path=feature_path,
        summary_path=summary_path,
    )

    assert (
        summary["artifacts"]["daily_sku_demand"]
        == daily_path.as_posix()
    )

    assert (
        summary["artifacts"]["forecast_features"]
        == feature_path.as_posix()
    )

    assert (
        summary["artifacts"]["summary"]
        == summary_path.as_posix()
    )


def test_materialization_is_reproducible(
    tmp_path: Path,
) -> None:
    """Repeated materialization should produce equivalent datasets."""
    raw = make_raw_transactions()

    first_feature_path = (
        tmp_path / "first_features.parquet"
    )
    second_feature_path = (
        tmp_path / "second_features.parquet"
    )

    first_summary = materialize_feature_pipeline(
        raw,
        daily_path=tmp_path / "first_daily.parquet",
        feature_path=first_feature_path,
        summary_path=tmp_path / "first_summary.json",
    )

    second_summary = materialize_feature_pipeline(
        raw,
        daily_path=tmp_path / "second_daily.parquet",
        feature_path=second_feature_path,
        summary_path=tmp_path / "second_summary.json",
    )

    first_features = pd.read_parquet(
        first_feature_path
    )
    second_features = pd.read_parquet(
        second_feature_path
    )

    pd.testing.assert_frame_equal(
        first_features,
        second_features,
    )

    assert (
        first_summary["feature_dataset"]
        == second_summary["feature_dataset"]
    )