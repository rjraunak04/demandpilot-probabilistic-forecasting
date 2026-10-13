from dataclasses import dataclass
from typing import Any

import pandas as pd

from demandpilot.evaluation.metrics import evaluate_forecast
from demandpilot.models.boosting import build_lightgbm, build_xgboost, fit_predict


@dataclass(frozen=True)
class ModelSelection:
    champion: str
    primary_metric: str
    validation_wape: float
    baseline_wape: float
    improvement_over_baseline_pct: float
    params: dict[str, Any]


TUNED_PARAMS: dict[str, dict[str, Any]] = {
    "lightgbm": {
        "n_estimators": 300,
        "learning_rate": 0.033743639849862894,
        "num_leaves": 48,
        "min_child_samples": 60,
        "colsample_bytree": 0.7366114704534336,
        "reg_lambda": 1.4855307303338106,
    },
    "xgboost": {
        "n_estimators": 200,
        "learning_rate": 0.042645919421477295,
        "max_depth": 7,
        "min_child_weight": 12,
        "subsample": 0.9462939903482535,
        "colsample_bytree": 0.7599021346475079,
        "reg_lambda": 1.5427033152408347,
    },
}


def tuned_estimator(model: str, *, random_state: int = 42):
    if model == "lightgbm":
        estimator = build_lightgbm(random_state=random_state)
    elif model == "xgboost":
        estimator = build_xgboost(random_state=random_state)
    else:
        raise ValueError(f"Unsupported model: {model}")
    estimator.set_params(**TUNED_PARAMS[model])
    return estimator


def evaluate_tuned_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for model in ("lightgbm", "xgboost"):
        prediction = fit_predict(tuned_estimator(model), train, validation)
        metrics = evaluate_forecast(validation["demand"], prediction)
        records.append({"model": model, "rows": len(validation), **metrics.as_dict()})
    return pd.DataFrame(records).sort_values(["wape", "mae", "model"]).reset_index(drop=True)


def select_champion(
    model_metrics: pd.DataFrame,
    *,
    baseline_wape: float,
) -> ModelSelection:
    required = {"model", "wape"}
    missing = required.difference(model_metrics.columns)
    if missing or model_metrics.empty:
        raise ValueError("Model metrics must contain model and wape.")

    winner = model_metrics.sort_values(["wape", "model"]).iloc[0]
    champion_wape = float(winner["wape"])
    improvement = 100.0 * (baseline_wape - champion_wape) / baseline_wape
    return ModelSelection(
        champion=str(winner["model"]),
        primary_metric="wape",
        validation_wape=champion_wape,
        baseline_wape=float(baseline_wape),
        improvement_over_baseline_pct=float(improvement),
        params=TUNED_PARAMS[str(winner["model"])].copy(),
    )
