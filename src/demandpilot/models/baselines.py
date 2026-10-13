from collections.abc import Callable

import numpy as np
import pandas as pd

BaselineFunction = Callable[[pd.DataFrame], np.ndarray]


def _validate_feature(frame: pd.DataFrame, column: str) -> np.ndarray:
    if frame.empty:
        raise ValueError("Forecast frame must not be empty.")
    if column not in frame.columns:
        raise ValueError(f"Required baseline feature is missing: {column}")

    values = frame[column].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Baseline feature contains non-finite values: {column}")
    if (values < 0).any():
        raise ValueError(f"Baseline feature contains negative demand: {column}")
    return values


def naive_forecast(frame: pd.DataFrame) -> np.ndarray:
    """Predict demand using the previous day's observed demand."""
    return _validate_feature(frame, "lag_1").copy()


def seasonal_naive_forecast(frame: pd.DataFrame) -> np.ndarray:
    """Predict demand using demand from the same weekday one week earlier."""
    return _validate_feature(frame, "lag_7").copy()


BASELINES: dict[str, BaselineFunction] = {
    "naive_lag_1": naive_forecast,
    "seasonal_naive_lag_7": seasonal_naive_forecast,
}
