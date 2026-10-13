import pandas as pd
import pytest

from demandpilot.evaluation.backtesting import (
    backtest_baselines,
    rolling_origin_folds,
    summarize_backtest,
)


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2011-10-15", "2011-11-11", freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "demand": [float(i % 5) for i in range(len(dates))],
            "lag_1": [float((i - 1) % 5) for i in range(len(dates))],
            "lag_7": [float((i - 7) % 5) for i in range(len(dates))],
        }
    )


def test_rolling_origin_builds_four_seven_day_folds() -> None:
    folds = rolling_origin_folds()
    assert len(folds) == 4
    assert all((fold.validation_end - fold.validation_start).days == 6 for fold in folds)


def test_fold_boundaries_are_contiguous_and_expanding() -> None:
    folds = rolling_origin_folds()
    assert folds[0].train_end == pd.Timestamp("2011-10-14")
    assert folds[-1].validation_end == pd.Timestamp("2011-11-11")
    for previous, current in zip(folds, folds[1:]):
        assert current.validation_start == previous.validation_end + pd.Timedelta(days=1)
        assert current.train_end == previous.validation_end


def test_partial_horizon_is_rejected() -> None:
    with pytest.raises(ValueError, match="complete forecast horizons"):
        rolling_origin_folds(validation_end="2011-11-10")


def test_non_positive_horizon_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        rolling_origin_folds(horizon_days=0)


def test_backtest_evaluates_every_fold_and_model() -> None:
    result = backtest_baselines(_frame())
    assert len(result) == 8
    assert result["fold"].nunique() == 4
    assert set(result["model"]) == {"naive_lag_1", "seasonal_naive_lag_7"}


def test_backtest_does_not_mutate_input() -> None:
    frame = _frame()
    original = frame.copy(deep=True)
    backtest_baselines(frame)
    pd.testing.assert_frame_equal(frame, original)


def test_backtest_requires_date() -> None:
    with pytest.raises(ValueError, match="date"):
        backtest_baselines(_frame().drop(columns=["date"]))


def test_backtest_rejects_empty_fold() -> None:
    frame = _frame().loc[lambda x: x["date"] <= "2011-10-21"]
    with pytest.raises(ValueError, match="contains no rows"):
        backtest_baselines(frame)


def test_summary_contains_four_folds_per_model() -> None:
    summary = summarize_backtest(backtest_baselines(_frame()))
    assert summary["folds"].eq(4).all()
    assert {"mean_wape", "std_wape", "mean_mae", "mean_rmse", "mean_smape"}.issubset(
        summary.columns
    )


def test_summary_is_ranked_by_mean_wape() -> None:
    summary = summarize_backtest(backtest_baselines(_frame()))
    assert summary["mean_wape"].is_monotonic_increasing
