import math

import numpy as np
import pytest

from demandpilot.evaluation.metrics import evaluate_forecast, mae, rmse, smape, wape


def test_perfect_forecast_has_zero_error() -> None:
    actual = [0, 1, 5, 10]
    predicted = [0, 1, 5, 10]

    metrics = evaluate_forecast(actual, predicted)

    assert metrics.wape == pytest.approx(0.0)
    assert metrics.mae == pytest.approx(0.0)
    assert metrics.rmse == pytest.approx(0.0)
    assert metrics.smape == pytest.approx(0.0)


def test_mae_matches_manual_calculation() -> None:
    assert mae([1, 2, 3], [2, 2, 5]) == pytest.approx(1.0)


def test_rmse_matches_manual_calculation() -> None:
    expected = math.sqrt((1 + 0 + 4) / 3)
    assert rmse([1, 2, 3], [2, 2, 5]) == pytest.approx(expected)


def test_wape_matches_manual_calculation() -> None:
    assert wape([10, 20, 30], [12, 18, 33]) == pytest.approx(7 / 60)


def test_smape_handles_joint_zero_safely() -> None:
    result = smape([0, 10], [0, 10])

    assert result == pytest.approx(0.0)


def test_all_zero_actual_and_prediction_returns_zero_wape() -> None:
    assert wape([0, 0], [0, 0]) == pytest.approx(0.0)


def test_all_zero_actual_with_error_has_undefined_wape() -> None:
    assert np.isnan(wape([0, 0], [1, 0]))


def test_negative_actual_demand_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        evaluate_forecast([1, -1], [1, 1])


def test_negative_predictions_are_supported_for_raw_model_evaluation() -> None:
    result = evaluate_forecast([0, 2], [-1, 2])

    assert result.mae == pytest.approx(0.5)


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        evaluate_forecast([1, 2], [1])


def test_empty_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_forecast([], [])


@pytest.mark.parametrize(
    ("actual", "predicted"),
    [
        ([1, np.nan], [1, 2]),
        ([1, 2], [1, np.inf]),
    ],
)
def test_non_finite_values_are_rejected(actual, predicted) -> None:
    with pytest.raises(ValueError, match="finite"):
        evaluate_forecast(actual, predicted)