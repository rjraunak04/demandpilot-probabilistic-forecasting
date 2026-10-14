"""Tests for leakage-safe demand lag features."""

import pandas as pd
import pytest

from demandpilot.features.lags import add_demand_lags


def make_panel() -> pd.DataFrame:
    """Create two independent continuous SKU demand histories."""
    dates = pd.date_range(
        "2010-01-01",
        periods=35,
        freq="D",
    )

    sku_a = pd.DataFrame(
        {
            "sku": "SKU-A",
            "date": dates,
            "demand": list(range(1, 36)),
            "is_observed_demand": True,
        }
    )

    sku_b = pd.DataFrame(
        {
            "sku": "SKU-B",
            "date": dates,
            "demand": list(range(101, 136)),
            "is_observed_demand": True,
        }
    )

    return pd.concat(
        [sku_a, sku_b],
        ignore_index=True,
    )


def test_lag_one_uses_previous_day_demand() -> None:
    panel = make_panel()

    featured, _ = add_demand_lags(panel)

    row = featured.loc[
        (featured["sku"] == "SKU-A")
        & (featured["date"] == pd.Timestamp("2010-01-02"))
    ].iloc[0]

    assert row["demand"] == 2
    assert row["lag_1"] == 1


def test_lag_seven_uses_seven_days_earlier() -> None:
    panel = make_panel()

    featured, _ = add_demand_lags(panel)

    row = featured.loc[
        (featured["sku"] == "SKU-A")
        & (featured["date"] == pd.Timestamp("2010-01-08"))
    ].iloc[0]

    assert row["demand"] == 8
    assert row["lag_7"] == 1


def test_lag_fourteen_uses_fourteen_days_earlier() -> None:
    panel = make_panel()

    featured, _ = add_demand_lags(panel)

    row = featured.loc[
        (featured["sku"] == "SKU-A")
        & (featured["date"] == pd.Timestamp("2010-01-15"))
    ].iloc[0]

    assert row["lag_14"] == 1


def test_lag_twenty_eight_uses_twenty_eight_days_earlier() -> None:
    panel = make_panel()

    featured, _ = add_demand_lags(panel)

    row = featured.loc[
        (featured["sku"] == "SKU-A")
        & (featured["date"] == pd.Timestamp("2010-01-29"))
    ].iloc[0]

    assert row["lag_28"] == 1


def test_lags_do_not_cross_sku_boundaries() -> None:
    panel = make_panel()

    featured, _ = add_demand_lags(panel)

    first_b = featured.loc[
        featured["sku"] == "SKU-B"
    ].iloc[0]

    assert pd.isna(first_b["lag_1"])
    assert pd.isna(first_b["lag_7"])
    assert pd.isna(first_b["lag_14"])
    assert pd.isna(first_b["lag_28"])


def test_future_mutation_does_not_change_historical_features() -> None:
    panel = make_panel()

    original_features, _ = add_demand_lags(panel)

    cutoff = pd.Timestamp("2010-01-20")

    mutated = panel.copy()
    future_mask = (
        (mutated["sku"] == "SKU-A")
        & (mutated["date"] > cutoff)
    )
    mutated.loc[future_mask, "demand"] = 999999

    mutated_features, _ = add_demand_lags(mutated)

    feature_columns = [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
    ]

    original_history = (
        original_features.loc[
            (original_features["sku"] == "SKU-A")
            & (original_features["date"] <= cutoff),
            feature_columns,
        ]
        .reset_index(drop=True)
    )

    mutated_history = (
        mutated_features.loc[
            (mutated_features["sku"] == "SKU-A")
            & (mutated_features["date"] <= cutoff),
            feature_columns,
        ]
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        original_history,
        mutated_history,
    )


def test_current_target_is_not_used_as_lag_one() -> None:
    panel = make_panel()

    featured, _ = add_demand_lags(panel)

    eligible = featured["lag_1"].notna()

    assert not (
        featured.loc[eligible, "lag_1"]
        == featured.loc[eligible, "demand"]
    ).all()


def test_initial_lag_history_remains_missing() -> None:
    panel = make_panel()

    featured, _ = add_demand_lags(panel)

    first_a = featured.loc[
        featured["sku"] == "SKU-A"
    ].iloc[0]

    assert pd.isna(first_a["lag_1"])
    assert pd.isna(first_a["lag_7"])
    assert pd.isna(first_a["lag_14"])
    assert pd.isna(first_a["lag_28"])


def test_lag_generation_preserves_row_count() -> None:
    panel = make_panel()

    featured, report = add_demand_lags(panel)

    assert len(featured) == len(panel)
    assert report.input_rows == len(panel)
    assert report.output_rows == len(panel)


def test_lag_generation_does_not_mutate_input() -> None:
    panel = make_panel()
    original = panel.copy(deep=True)

    add_demand_lags(panel)

    pd.testing.assert_frame_equal(
        panel,
        original,
    )


def test_lag_generation_rejects_calendar_gaps() -> None:
    panel = make_panel()

    panel = panel.loc[
        ~(
            (panel["sku"] == "SKU-A")
            & (panel["date"] == pd.Timestamp("2010-01-10"))
        )
    ].reset_index(drop=True)

    with pytest.raises(
        ValueError,
        match="continuous daily calendar",
    ):
        add_demand_lags(panel)


def test_lag_generation_rejects_invalid_lags() -> None:
    panel = make_panel()

    with pytest.raises(
        ValueError,
        match="positive integers",
    ):
        add_demand_lags(
            panel,
            lags=(0, 7),
        )


def test_lag_generation_rejects_duplicate_lags() -> None:
    panel = make_panel()

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        add_demand_lags(
            panel,
            lags=(1, 7, 7),
        )