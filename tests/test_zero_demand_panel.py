"""Tests for continuous SKU-level demand panel reconstruction."""

import pandas as pd
import pytest

from demandpilot.features.panel import reconstruct_zero_demand_days


def make_daily_demand() -> pd.DataFrame:
    """Create observed SKU-day demand with intentional calendar gaps."""
    return pd.DataFrame(
        {
            "sku": [
                "SKU-A",
                "SKU-A",
                "SKU-A",
                "SKU-B",
                "SKU-B",
            ],
            "date": [
                "2010-01-02",
                "2010-01-04",
                "2010-01-05",
                "2010-01-01",
                "2010-01-03",
            ],
            "demand": [
                5,
                7,
                2,
                10,
                4,
            ],
        }
    )


def test_panel_fills_internal_missing_days_with_zero() -> None:
    daily = make_daily_demand()

    panel, _ = reconstruct_zero_demand_days(daily)

    row = panel.loc[
        (panel["sku"] == "SKU-A")
        & (panel["date"] == pd.Timestamp("2010-01-03"))
    ]

    assert len(row) == 1
    assert row.iloc[0]["demand"] == 0
    assert not bool(row.iloc[0]["is_observed_demand"])


def test_panel_preserves_observed_demand() -> None:
    daily = make_daily_demand()

    panel, _ = reconstruct_zero_demand_days(daily)

    row = panel.loc[
        (panel["sku"] == "SKU-A")
        & (panel["date"] == pd.Timestamp("2010-01-04"))
    ]

    assert len(row) == 1
    assert row.iloc[0]["demand"] == 7
    assert bool(row.iloc[0]["is_observed_demand"])


def test_panel_does_not_create_pre_launch_zeros() -> None:
    daily = make_daily_demand()

    panel, _ = reconstruct_zero_demand_days(daily)

    sku_a = panel.loc[
        panel["sku"] == "SKU-A"
    ]

    assert sku_a["date"].min() == pd.Timestamp("2010-01-02")


def test_panel_does_not_create_post_last_sale_zeros() -> None:
    daily = make_daily_demand()

    panel, _ = reconstruct_zero_demand_days(daily)

    sku_b = panel.loc[
        panel["sku"] == "SKU-B"
    ]

    assert sku_b["date"].max() == pd.Timestamp("2010-01-03")


def test_panel_is_continuous_within_each_sku() -> None:
    daily = make_daily_demand()

    panel, _ = reconstruct_zero_demand_days(daily)

    for _, sku_data in panel.groupby(
        "sku",
        observed=True,
    ):
        expected_dates = pd.date_range(
            sku_data["date"].min(),
            sku_data["date"].max(),
            freq="D",
        )

        assert list(sku_data["date"]) == list(expected_dates)


def test_panel_has_unique_sku_day_grain() -> None:
    daily = make_daily_demand()

    panel, _ = reconstruct_zero_demand_days(daily)

    assert not panel.duplicated(["sku", "date"]).any()


def test_panel_preserves_total_demand() -> None:
    daily = make_daily_demand()

    panel, report = reconstruct_zero_demand_days(daily)

    assert panel["demand"].sum() == daily["demand"].sum()
    assert report.total_demand_before == report.total_demand_after


def test_panel_keeps_sku_calendars_independent() -> None:
    daily = make_daily_demand()

    panel, _ = reconstruct_zero_demand_days(daily)

    sku_a = panel.loc[panel["sku"] == "SKU-A"]
    sku_b = panel.loc[panel["sku"] == "SKU-B"]

    assert len(sku_a) == 4
    assert len(sku_b) == 3


def test_panel_report_reconciles_inserted_rows() -> None:
    daily = make_daily_demand()

    panel, report = reconstruct_zero_demand_days(daily)

    assert report.observed_rows == len(daily)
    assert report.panel_rows == len(panel)
    assert (
        report.inserted_zero_rows
        == report.panel_rows - report.observed_rows
    )


def test_panel_rejects_duplicate_input_grain() -> None:
    daily = make_daily_demand()

    duplicate = daily.iloc[[0]].copy()
    daily = pd.concat(
        [daily, duplicate],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="unique SKU-day grain",
    ):
        reconstruct_zero_demand_days(daily)


def test_panel_rejects_non_positive_observed_demand() -> None:
    daily = make_daily_demand()
    daily.loc[0, "demand"] = 0

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        reconstruct_zero_demand_days(daily)


def test_panel_does_not_mutate_input() -> None:
    daily = make_daily_demand()
    original = daily.copy(deep=True)

    reconstruct_zero_demand_days(daily)

    pd.testing.assert_frame_equal(
        daily,
        original,
    )