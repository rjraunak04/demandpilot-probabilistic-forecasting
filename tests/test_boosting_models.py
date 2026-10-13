import numpy as np
import pandas as pd
import pytest

from demandpilot.models.boosting import MODEL_FEATURES, feature_matrix, fit_predict, target_vector


def _frame(rows: int = 8) -> pd.DataFrame:
    data = {name: np.arange(rows, dtype=float) + 1 for name in MODEL_FEATURES}
    data["demand"] = np.arange(rows, dtype=float)
    return pd.DataFrame(data)


def test_feature_matrix_uses_locked_feature_contract() -> None:
    matrix = feature_matrix(_frame())
    assert tuple(matrix.columns) == MODEL_FEATURES
    assert "demand" not in matrix.columns


def test_feature_matrix_rejects_missing_feature() -> None:
    with pytest.raises(ValueError, match="Missing model features"):
        feature_matrix(_frame().drop(columns=[MODEL_FEATURES[0]]))


def test_feature_matrix_rejects_non_finite_values() -> None:
    frame = _frame()
    frame.loc[0, MODEL_FEATURES[0]] = np.nan
    with pytest.raises(ValueError, match="finite and complete"):
        feature_matrix(frame)


def test_target_vector_accepts_non_negative_demand() -> None:
    np.testing.assert_array_equal(target_vector(_frame()), np.arange(8, dtype=float))


def test_target_vector_rejects_negative_demand() -> None:
    frame = _frame()
    frame.loc[0, "demand"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        target_vector(frame)


class _DummyRegressor:
    def fit(self, x, y):
        self.rows = len(x)
        return self

    def predict(self, x):
        return np.array([-1.0] + [2.0] * (len(x) - 1))


def test_fit_predict_clips_negative_demand_forecasts() -> None:
    prediction = fit_predict(_DummyRegressor(), _frame(8), _frame(3))
    assert prediction.shape == (3,)
    assert prediction[0] == 0.0
    assert (prediction >= 0).all()
