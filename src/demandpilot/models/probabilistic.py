"""Probabilistic LightGBM models for lead-time demand."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from demandpilot.models.boosting import feature_matrix

QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class QuantileForecast:
    p10: np.ndarray
    p50: np.ndarray
    p90: np.ndarray

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"p10": self.p10, "p50": self.p50, "p90": self.p90})


def pinball_loss(y_true, y_pred, quantile: float) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between zero and one.")
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Pinball inputs must be non-empty with identical shapes.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Pinball inputs must be finite.")
    error = actual - predicted
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def build_quantile_lightgbm(quantile: float, *, random_state: int = 42):
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between zero and one.")
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError("Install project dependencies to use LightGBM.") from exc
    return LGBMRegressor(
        objective="quantile",
        alpha=quantile,
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=48,
        min_child_samples=60,
        colsample_bytree=0.75,
        reg_lambda=1.5,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def _lead_target(frame: pd.DataFrame) -> np.ndarray:
    if "lead_time_demand" not in frame:
        raise ValueError("Frame must contain lead_time_demand.")
    target = frame["lead_time_demand"].to_numpy(dtype=float)
    if target.size == 0 or not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("Lead-time target must be finite, non-negative, and non-empty.")
    return target


def fit_quantile_forecast(
    train: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    *,
    random_state: int = 42,
) -> QuantileForecast:
    x_train = feature_matrix(train)
    y_train = _lead_target(train)
    x_pred = feature_matrix(prediction_frame)
    predictions: dict[float, np.ndarray] = {}
    for quantile in QUANTILES:
        model = build_quantile_lightgbm(quantile, random_state=random_state)
        model.fit(x_train, y_train)
        raw = np.clip(np.asarray(model.predict(x_pred), dtype=float), 0.0, None)
        if raw.shape != (len(prediction_frame),) or not np.isfinite(raw).all():
            raise ValueError("Quantile model produced invalid predictions.")
        predictions[quantile] = raw

    stacked = np.vstack([predictions[q] for q in QUANTILES])
    ordered = np.sort(stacked, axis=0)
    return QuantileForecast(p10=ordered[0], p50=ordered[1], p90=ordered[2])


def quantile_metrics(actual, forecast: QuantileForecast) -> dict[str, float]:
    y = np.asarray(actual, dtype=float)
    coverage_80 = float(np.mean((y >= forecast.p10) & (y <= forecast.p90)))
    width = float(np.mean(forecast.p90 - forecast.p10))
    return {
        "pinball_p10": pinball_loss(y, forecast.p10, 0.1),
        "pinball_p50": pinball_loss(y, forecast.p50, 0.5),
        "pinball_p90": pinball_loss(y, forecast.p90, 0.9),
        "coverage_80": coverage_80,
        "mean_interval_width": width,
    }
