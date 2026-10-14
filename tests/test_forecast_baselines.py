import numpy as np
import pandas as pd
import pytest

from demandpilot.evaluation.benchmark import benchmark_baselines
from demandpilot.models.baselines import naive_forecast, seasonal_naive_forecast


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "demand": [0.0, 5.0, 2.0],
            "lag_1": [1.0, 4.0, 2.0],
            "lag_7": [0.0, 5.0, 1.0],
        }
    )


def test_naive_uses_lag_1() -> None:
    np.testing.assert_array_equal(naive_forecast(_frame()), [1.0, 4.0, 2.0])


def test_seasonal_naive_uses_lag_7() -> None:
    np.testing.assert_array_equal(seasonal_naive_forecast(_frame()), [0.0, 5.0, 1.0])


def test_baselines_do_not_mutate_input() -> None:
    frame = _frame()
    original = frame.copy(deep=True)
    naive_forecast(frame)
    seasonal_naive_forecast(frame)
    pd.testing.assert_frame_equal(frame, original)


def test_missing_lag_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing"):
        naive_forecast(_frame().drop(columns=["lag_1"]))


def test_non_finite_lag_is_rejected() -> None:
    frame = _frame()
    frame.loc[0, "lag_1"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        naive_forecast(frame)


def test_negative_historical_demand_is_rejected() -> None:
    frame = _frame()
    frame.loc[0, "lag_7"] = -1.0
    with pytest.raises(ValueError, match="negative"):
        seasonal_naive_forecast(frame)


def test_empty_frame_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        naive_forecast(pd.DataFrame())


def test_benchmark_contains_both_baselines() -> None:
    result = benchmark_baselines(_frame())
    assert set(result["model"]) == {"naive_lag_1", "seasonal_naive_lag_7"}


def test_benchmark_reports_expected_metrics() -> None:
    result = benchmark_baselines(_frame())
    assert {"wape", "mae", "rmse", "smape"}.issubset(result.columns)
    assert result["rows"].eq(3).all()


def test_benchmark_is_sorted_by_wape() -> None:
    assert benchmark_baselines(_frame())["wape"].is_monotonic_increasing


def test_benchmark_requires_demand() -> None:
    with pytest.raises(ValueError, match="demand"):
        benchmark_baselines(_frame().drop(columns=["demand"]))
