from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastMetrics:
    wape: float
    mae: float
    rmse: float
    smape: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _validated_arrays(
    y_true: pd.Series | np.ndarray | list[float],
    y_pred: pd.Series | np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional.")

    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have identical shapes.")

    if actual.size == 0:
        raise ValueError("Forecast metric inputs must not be empty.")

    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Forecast metric inputs must contain only finite values.")

    if (actual < 0).any():
        raise ValueError("Observed demand must be non-negative.")

    return actual, predicted


def mae(y_true, y_pred) -> float:
    actual, predicted = _validated_arrays(y_true, y_pred)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(y_true, y_pred) -> float:
    actual, predicted = _validated_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def wape(y_true, y_pred) -> float:
    actual, predicted = _validated_arrays(y_true, y_pred)

    denominator = float(np.sum(np.abs(actual)))
    if denominator == 0.0:
        return 0.0 if np.allclose(actual, predicted) else float("nan")

    return float(np.sum(np.abs(actual - predicted)) / denominator)


def smape(y_true, y_pred) -> float:
    actual, predicted = _validated_arrays(y_true, y_pred)

    denominator = np.abs(actual) + np.abs(predicted)
    numerator = 2.0 * np.abs(actual - predicted)

    contributions = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )

    return float(np.mean(contributions))


def evaluate_forecast(y_true, y_pred) -> ForecastMetrics:
    actual, predicted = _validated_arrays(y_true, y_pred)

    return ForecastMetrics(
        wape=wape(actual, predicted),
        mae=mae(actual, predicted),
        rmse=rmse(actual, predicted),
        smape=smape(actual, predicted),
    )