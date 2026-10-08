"""Integration tests for the complete forecasting feature pipeline."""

import pandas as pd
import pytest

from demandpilot.features.pipeline import build_forecasting_features
from demandpilot.features.validate import (
    expected_feature_columns,
    validate_forecasting_features,
)


def make_raw_transactions() -> pd.DataFrame:
    """Create deterministic raw transactions spanning enough history."""
    rows: list[dict[str, object]] = []

    dates = pd.date_range(
        "2010-01-01",
        periods=40,
        freq="D",
    )

    for sku_index, sku in enumerate(
        ("SKU-A", "SKU-B")
    ):
        for day_index, date in enumerate(dates):
            # Intentional zero-demand days are represented by no transaction.
            if day_index in {5, 12, 20}:
                continue

            rows.append(
                {
                    "Invoice": (
                        f"{sku_index + 1}"
                        f"{day_index:05d}"
                    ),
                    "StockCode": sku,
                    "Description": f"{sku} product",
                    "Quantity": (
                        day_index + 1
                        if sku == "SKU-A"
                        else day_index + 101
                    ),
                    "InvoiceDate": (
                        date + pd.Timedelta(hours=10)
                    ),
                    "Price": 2.5,
                    "Customer ID": 1000 + sku_index,
                    "Country": "United Kingdom",
                    "SourceSheet": "Year 2009-2010",
                }
            )

    return pd.DataFrame(rows)


def test_end_to_end_pipeline_satisfies_feature_contract() -> None:
    """Complete pipeline output should satisfy the feature contract."""
    raw = make_raw_transactions()

    features, _ = build_forecasting_features(raw)

    validate_forecasting_features(features)

    assert expected_feature_columns().issubset(
        features.columns
    )


def test_end_to_end_pipeline_reconstructs_zero_demand_days() -> None:
    """Missing transaction days should become explicit zero-demand days."""
    raw = make_raw_transactions()

    features, report = build_forecasting_features(raw)

    assert report.inserted_zero_rows == 6

    reconstructed = features.loc[
        ~features["is_observed_demand"]
    ]

    assert len(reconstructed) == 6
    assert (reconstructed["demand"] == 0).all()


def test_pipeline_is_deterministic() -> None:
    """Repeated runs on identical input should produce identical output."""
    raw = make_raw_transactions()

    first, first_report = build_forecasting_features(
        raw
    )
    second, second_report = build_forecasting_features(
        raw
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )

    assert (
        first_report.as_dict()
        == second_report.as_dict()
    )


def test_pipeline_is_independent_of_input_row_order() -> None:
    """Source transaction ordering should not affect final features."""
    raw = make_raw_transactions()

    baseline, _ = build_forecasting_features(raw)

    shuffled = raw.sample(
        frac=1.0,
        random_state=42,
    ).reset_index(drop=True)

    shuffled_features, _ = build_forecasting_features(
        shuffled
    )

    pd.testing.assert_frame_equal(
        baseline,
        shuffled_features,
    )


def test_future_mutation_does_not_change_historical_features() -> None:
    """Future demand changes must not alter historical feature values."""
    raw = make_raw_transactions()

    baseline, _ = build_forecasting_features(raw)

    cutoff = pd.Timestamp("2010-01-25")

    mutated = raw.copy()

    future_mask = (
        (mutated["StockCode"] == "SKU-A")
        & (
            pd.to_datetime(
                mutated["InvoiceDate"]
            )
            > cutoff
        )
    )

    mutated.loc[
        future_mask,
        "Quantity",
    ] = 999999

    changed, _ = build_forecasting_features(
        mutated
    )

    feature_columns = [
        column
        for column in baseline.columns
        if column
        not in {
            "demand",
            "is_observed_demand",
        }
    ]

    baseline_history = (
        baseline.loc[
            (baseline["sku"] == "SKU-A")
            & (baseline["date"] <= cutoff),
            feature_columns,
        ]
        .reset_index(drop=True)
    )

    changed_history = (
        changed.loc[
            (changed["sku"] == "SKU-A")
            & (changed["date"] <= cutoff),
            feature_columns,
        ]
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        baseline_history,
        changed_history,
    )


def test_current_target_does_not_enter_same_day_history_features() -> None:
    """Changing demand(t) must not change historical features at t."""
    raw = make_raw_transactions()

    baseline, _ = build_forecasting_features(raw)

    target_date = pd.Timestamp("2010-01-25")

    mutated = raw.copy()

    target_mask = (
        (mutated["StockCode"] == "SKU-A")
        & (
            pd.to_datetime(
                mutated["InvoiceDate"]
            ).dt.normalize()
            == target_date
        )
    )

    mutated.loc[
        target_mask,
        "Quantity",
    ] = 999999

    changed, _ = build_forecasting_features(
        mutated
    )

    historical_feature_columns = [
        column
        for column in baseline.columns
        if (
            column.startswith("lag_")
            or column.startswith("rolling_")
        )
    ]

    baseline_row = (
        baseline.loc[
            (baseline["sku"] == "SKU-A")
            & (baseline["date"] == target_date),
            historical_feature_columns,
        ]
        .reset_index(drop=True)
    )

    changed_row = (
        changed.loc[
            (changed["sku"] == "SKU-A")
            & (changed["date"] == target_date),
            historical_feature_columns,
        ]
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        baseline_row,
        changed_row,
    )


def test_mutating_one_sku_does_not_change_other_sku_features() -> None:
    """Feature calculations must remain isolated between SKUs."""
    raw = make_raw_transactions()

    baseline, _ = build_forecasting_features(raw)

    mutated = raw.copy()

    mutated.loc[
        mutated["StockCode"] == "SKU-A",
        "Quantity",
    ] *= 100

    changed, _ = build_forecasting_features(
        mutated
    )

    baseline_b = (
        baseline.loc[
            baseline["sku"] == "SKU-B"
        ]
        .reset_index(drop=True)
    )

    changed_b = (
        changed.loc[
            changed["sku"] == "SKU-B"
        ]
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        baseline_b,
        changed_b,
    )


def test_complete_history_begins_after_maximum_history_window() -> None:
    """Full 28-day features should start only after 28 past days."""
    raw = make_raw_transactions()

    features, report = build_forecasting_features(
        raw
    )

    complete_columns = [
        "lag_28",
        "rolling_mean_28",
        "rolling_std_28",
        "rolling_min_28",
        "rolling_max_28",
    ]

    complete = (
        features[complete_columns]
        .notna()
        .all(axis=1)
    )

    assert (
        int(complete.sum())
        == report.complete_history_rows
    )

    for _, sku_data in features.groupby(
        "sku",
        observed=True,
    ):
        sku_complete = (
            sku_data[complete_columns]
            .notna()
            .all(axis=1)
            .reset_index(drop=True)
        )

        # Rows 0-27 have fewer than 28 prior calendar days.
        assert not sku_complete.iloc[:28].any()

        # Row 28 has exactly 28 prior calendar days.
        if len(sku_complete) > 28:
            assert sku_complete.iloc[28]


def test_pipeline_preserves_non_negative_target() -> None:
    """Final target demand must remain non-negative."""
    raw = make_raw_transactions()

    features, _ = build_forecasting_features(raw)

    assert (features["demand"] >= 0).all()


def test_pipeline_output_has_unique_sku_day_grain() -> None:
    """Final feature output must contain one row per SKU-day."""
    raw = make_raw_transactions()

    features, _ = build_forecasting_features(raw)

    assert not features.duplicated(
        ["sku", "date"]
    ).any()


def test_feature_validator_rejects_duplicate_grain() -> None:
    """Feature contract should reject duplicate SKU-day rows."""
    raw = make_raw_transactions()

    features, _ = build_forecasting_features(raw)

    duplicate = features.iloc[[0]].copy()

    invalid = pd.concat(
        [features, duplicate],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate SKU-day",
    ):
        validate_forecasting_features(
            invalid
        )


def test_feature_validator_rejects_missing_feature() -> None:
    """Feature contract should reject a missing required feature."""
    raw = make_raw_transactions()

    features, _ = build_forecasting_features(raw)

    invalid = features.drop(
        columns=["lag_28"]
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        validate_forecasting_features(
            invalid
        )


def test_feature_validator_rejects_negative_demand() -> None:
    """Feature contract should reject negative target demand."""
    raw = make_raw_transactions()

    features, _ = build_forecasting_features(raw)

    invalid = features.copy()
    invalid.loc[
        invalid.index[0],
        "demand",
    ] = -1

    with pytest.raises(
        ValueError,
        match="negative demand",
    ):
        validate_forecasting_features(
            invalid
        )


def test_feature_validator_rejects_calendar_gap() -> None:
    """Feature contract should reject gaps inside an SKU lifetime."""
    raw = make_raw_transactions()

    features, _ = build_forecasting_features(raw)

    sku_a_indices = features.index[
        features["sku"] == "SKU-A"
    ]

    invalid = features.drop(
        index=sku_a_indices[10]
    ).reset_index(drop=True)

    with pytest.raises(
        ValueError,
        match="calendar gaps",
    ):
        validate_forecasting_features(
            invalid
        )


def test_pipeline_report_reconciles_rows() -> None:
    """Pipeline audit counts should reconcile end to end."""
    raw = make_raw_transactions()

    features, report = build_forecasting_features(
        raw
    )

    assert report.raw_rows == len(raw)
    assert report.feature_rows == len(features)

    assert (
        report.complete_history_rows
        + report.incomplete_history_rows
        == report.feature_rows
    )

    assert (
        report.inserted_zero_rows
        == report.panel_rows
        - report.observed_daily_rows
    )


def test_complete_history_share_is_bounded() -> None:
    """Complete-history share should be a valid proportion."""
    raw = make_raw_transactions()

    _, report = build_forecasting_features(
        raw
    )

    assert 0.0 <= report.complete_history_share <= 1.0