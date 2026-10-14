"""Tests for leakage-safe temporal forecasting features."""

import pandas as pd
import pytest

from demandpilot.features.temporal import add_temporal_features


def make_panel() -> pd.DataFrame:
    """Create two continuous SKU histories."""
    dates = pd.date_range(
        "2010-01-01",
        periods=40,
        freq="D",
    )

    sku_a = pd.DataFrame(
        {
            "sku": "SKU-A",
            "date": dates,
            "demand": list(range(1, 41)),
        }
    )

    sku_b = pd.DataFrame(
        {
            "sku": "SKU-B",
            "date": dates,
            "demand": list(range(101, 141)),
        }
    )

    return pd.concat(
        [sku_a, sku_b],
        ignore_index=True,
    )


def test_rolling_mean_uses_only_previous_days() -> None:
    panel = make_panel()

    featured, _ = add_temporal_features(panel)

    row = featured.loc[
        (featured["sku"] == "SKU-A")
        & (featured["date"] == pd.Timestamp("2010-01-08"))
    ].iloc[0]

    assert row["demand"] == 8
    assert row["rolling_mean_7"] == 4


def test_rolling_min_and_max_use_past_only() -> None:
    panel = make_panel()

    featured, _ = add_temporal_features(panel)

    row = featured.loc[
        (featured["sku"] == "SKU-A")
        & (featured["date"] == pd.Timestamp("2010-01-08"))
    ].iloc[0]

    assert row["rolling_min_7"] == 1
    assert row["rolling_max_7"] == 7


def test_rolling_std_uses_past_only_window() -> None:
    panel = make_panel()

    featured, _ = add_temporal_features(panel)

    expected = pd.Series(
        range(1, 8),
        dtype="float64",
    ).std(ddof=0)

    row = featured.loc[
        (featured["sku"] == "SKU-A")
        & (featured["date"] == pd.Timestamp("2010-01-08"))
    ].iloc[0]

    assert row["rolling_std_7"] == pytest.approx(expected)


def test_rolling_features_do_not_cross_skus() -> None:
    panel = make_panel()

    featured, _ = add_temporal_features(panel)

    first_b = featured.loc[
        featured["sku"] == "SKU-B"
    ].iloc[0]

    assert pd.isna(first_b["rolling_mean_7"])
    assert pd.isna(first_b["rolling_mean_14"])
    assert pd.isna(first_b["rolling_mean_28"])


def test_future_mutation_does_not_change_historical_features() -> None:
    panel = make_panel()

    original, _ = add_temporal_features(panel)

    cutoff = pd.Timestamp("2010-01-20")

    mutated = panel.copy()

    future_mask = (
        (mutated["sku"] == "SKU-A")
        & (mutated["date"] > cutoff)
    )

    mutated.loc[
        future_mask,
        "demand",
    ] = 999999

    changed, _ = add_temporal_features(mutated)

    columns = [
        "rolling_mean_7",
        "rolling_std_7",
        "rolling_mean_14",
        "rolling_std_14",
    ]

    original_history = (
        original.loc[
            (original["sku"] == "SKU-A")
            & (original["date"] <= cutoff),
            columns,
        ]
        .reset_index(drop=True)
    )

    changed_history = (
        changed.loc[
            (changed["sku"] == "SKU-A")
            & (changed["date"] <= cutoff),
            columns,
        ]
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        original_history,
        changed_history,
    )


def test_current_target_mutation_does_not_change_same_day_features() -> None:
    panel = make_panel()

    target_date = pd.Timestamp("2010-01-20")

    original, _ = add_temporal_features(panel)

    mutated = panel.copy()

    mask = (
        (mutated["sku"] == "SKU-A")
        & (mutated["date"] == target_date)
    )

    mutated.loc[mask, "demand"] = 999999

    changed, _ = add_temporal_features(mutated)

    columns = [
        "rolling_mean_7",
        "rolling_std_7",
        "rolling_min_7",
        "rolling_max_7",
        "rolling_mean_14",
        "rolling_mean_28",
    ]

    original_row = (
        original.loc[
            (original["sku"] == "SKU-A")
            & (original["date"] == target_date),
            columns,
        ]
        .reset_index(drop=True)
    )

    changed_row = (
        changed.loc[
            (changed["sku"] == "SKU-A")
            & (changed["date"] == target_date),
            columns,
        ]
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        original_row,
        changed_row,
    )


def test_calendar_features_are_correct() -> None:
    panel = make_panel()

    featured, _ = add_temporal_features(panel)

    row = featured.loc[
        (featured["sku"] == "SKU-A")
        & (featured["date"] == pd.Timestamp("2010-01-02"))
    ].iloc[0]

    assert row["day_of_week"] == 5
    assert row["day_of_month"] == 2
    assert row["month"] == 1
    assert row["quarter"] == 1
    assert row["is_weekend"] == 1


def test_weekday_is_not_marked_as_weekend() -> None:
    panel = make_panel()

    featured, _ = add_temporal_features(panel)

    row = featured.loc[
        (featured["sku"] == "SKU-A")
        & (featured["date"] == pd.Timestamp("2010-01-04"))
    ].iloc[0]

    assert row["day_of_week"] == 0
    assert row["is_weekend"] == 0


def test_rolling_history_is_missing_until_window_available() -> None:
    panel = make_panel()

    featured, _ = add_temporal_features(panel)

    sku_a = featured.loc[
        featured["sku"] == "SKU-A"
    ].reset_index(drop=True)

    assert sku_a.loc[6, "rolling_mean_7"] != (
        sku_a.loc[6, "rolling_mean_7"]
    )
    assert sku_a.loc[7, "rolling_mean_7"] == 4


def test_temporal_features_preserve_rows() -> None:
    panel = make_panel()

    featured, report = add_temporal_features(panel)

    assert len(featured) == len(panel)
    assert report.input_rows == len(panel)
    assert report.output_rows == len(panel)


def test_temporal_features_do_not_mutate_input() -> None:
    panel = make_panel()
    original = panel.copy(deep=True)

    add_temporal_features(panel)

    pd.testing.assert_frame_equal(
        panel,
        original,
    )


def test_temporal_features_reject_calendar_gaps() -> None:
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
        add_temporal_features(panel)


def test_temporal_features_reject_invalid_window() -> None:
    panel = make_panel()

    with pytest.raises(
        ValueError,
        match="positive integers",
    ):
        add_temporal_features(
            panel,
            windows=(0, 7),
        )


def test_temporal_features_reject_duplicate_windows() -> None:
    panel = make_panel()

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        add_temporal_features(
            panel,
            windows=(7, 7),
        )