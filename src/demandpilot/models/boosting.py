from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET = "demand"
EXCLUDED_FEATURES = {"sku", "date", "demand", "is_observed_demand"}
MODEL_FEATURES = (
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_min_7",
    "rolling_max_7",
    "rolling_mean_14",
    "rolling_std_14",
    "rolling_min_14",
    "rolling_max_14",
    "rolling_mean_28",
    "rolling_std_28",
    "rolling_min_28",
    "rolling_max_28",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "is_weekend",
)


@dataclass(frozen=True)
class BoostingSpec:
    name: str
    estimator: object


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(MODEL_FEATURES).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")
    matrix = frame.loc[:, MODEL_FEATURES].astype(float)
    if matrix.isna().any().any() or not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("Model features must be finite and complete.")
    return matrix


def target_vector(frame: pd.DataFrame) -> np.ndarray:
    if TARGET not in frame.columns:
        raise ValueError("Model frame must contain demand.")
    target = frame[TARGET].to_numpy(dtype=float)
    if target.size == 0 or not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("Demand target must be finite, non-negative, and non-empty.")
    return target


def build_lightgbm(*, random_state: int = 42):
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError("Install project dependencies to use LightGBM.") from exc

    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def build_xgboost(*, random_state: int = 42):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("Install project dependencies to use XGBoost.") from exc

    return XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=350,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=10,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )


def model_specs(*, random_state: int = 42) -> tuple[BoostingSpec, ...]:
    return (
        BoostingSpec("lightgbm", build_lightgbm(random_state=random_state)),
        BoostingSpec("xgboost", build_xgboost(random_state=random_state)),
    )


def fit_predict(estimator, train: pd.DataFrame, validation: pd.DataFrame) -> np.ndarray:
    estimator.fit(feature_matrix(train), target_vector(train))
    prediction = np.asarray(estimator.predict(feature_matrix(validation)), dtype=float)
    if prediction.shape != (len(validation),) or not np.isfinite(prediction).all():
        raise ValueError("Model produced invalid predictions.")
    return np.clip(prediction, 0.0, None)
