from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from demandpilot.evaluation.metrics import wape
from demandpilot.models.boosting import feature_matrix, target_vector


@dataclass(frozen=True)
class TuningResult:
    model: str
    best_value: float
    best_params: dict[str, Any]
    trials: int


def _tuning_window(
    train: pd.DataFrame,
    *,
    validation_days: int = 7,
    training_days: int = 180,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train.empty or "date" not in train.columns:
        raise ValueError("Training frame must be non-empty and contain date.")
    if validation_days <= 0 or training_days <= 0:
        raise ValueError("Tuning window lengths must be positive.")

    data = train.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    validation_end = data["date"].max()
    validation_start = validation_end - timedelta(days=validation_days - 1)
    training_end = validation_start - timedelta(days=1)
    training_start = training_end - timedelta(days=training_days - 1)

    fit = data.loc[data["date"].between(training_start, training_end)].copy()
    validation = data.loc[data["date"].between(validation_start, validation_end)].copy()
    if fit.empty or validation.empty or fit["date"].max() >= validation["date"].min():
        raise ValueError("Unable to construct leakage-safe tuning window.")
    return fit, validation


def tune_model(
    train: pd.DataFrame,
    *,
    model: str,
    trials: int = 8,
    random_state: int = 42,
) -> TuningResult:
    if trials <= 0:
        raise ValueError("trials must be positive.")
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Install project dependencies to use Optuna.") from exc

    fit, validation = _tuning_window(train)
    x_train, y_train = feature_matrix(fit), target_vector(fit)
    x_valid, y_valid = feature_matrix(validation), target_vector(validation)

    builders: dict[str, Callable] = {
        "lightgbm": _lightgbm_trial,
        "xgboost": _xgboost_trial,
    }
    if model not in builders:
        raise ValueError(f"Unsupported tuning model: {model}")

    def objective(trial) -> float:
        estimator = builders[model](trial, random_state)
        estimator.fit(x_train, y_train)
        prediction = np.clip(np.asarray(estimator.predict(x_valid), dtype=float), 0.0, None)
        return wape(y_valid, prediction)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    return TuningResult(model, float(study.best_value), dict(study.best_params), len(study.trials))


def _lightgbm_trial(trial, random_state: int):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=trial.suggest_int("n_estimators", 200, 500, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.03, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 63),
        min_child_samples=trial.suggest_int("min_child_samples", 30, 100, step=10),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.0, 3.0),
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def _xgboost_trial(trial, random_state: int):
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=trial.suggest_int("n_estimators", 200, 500, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.03, 0.1, log=True),
        max_depth=trial.suggest_int("max_depth", 5, 10),
        min_child_weight=trial.suggest_int("min_child_weight", 5, 20),
        subsample=trial.suggest_float("subsample", 0.75, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.0, 3.0),
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )
