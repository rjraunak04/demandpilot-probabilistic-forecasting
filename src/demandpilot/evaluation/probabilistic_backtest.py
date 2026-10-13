"""Rolling-origin probabilistic evaluation for 7-day lead-time demand."""

from __future__ import annotations

import numpy as np
import pandas as pd

from demandpilot.evaluation.backtesting import rolling_origin_folds
from demandpilot.evaluation.lead_time import snapshot_examples, training_examples_at_origin
from demandpilot.evaluation.metrics import evaluate_forecast
from demandpilot.models.probabilistic import fit_quantile_forecast, quantile_metrics


def probabilistic_backtest(examples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_records: list[dict[str, float | int | str]] = []
    predictions: list[pd.DataFrame] = []

    for fold in rolling_origin_folds():
        origin = fold.validation_start
        train = training_examples_at_origin(examples, origin)
        snapshot = snapshot_examples(examples, origin)
        forecast = fit_quantile_forecast(train, snapshot, random_state=42 + fold.fold)
        actual = snapshot["lead_time_demand"].to_numpy(dtype=float)
        point = evaluate_forecast(actual, forecast.p50)
        probability = quantile_metrics(actual, forecast)

        seasonal_total = snapshot["rolling_mean_7"].to_numpy(dtype=float) * 7.0
        baseline = evaluate_forecast(actual, seasonal_total)

        fold_records.append(
            {
                "fold": fold.fold,
                "origin": str(origin.date()),
                "train_rows": len(train),
                "forecast_rows": len(snapshot),
                "p50_wape": point.wape,
                "seasonal_7d_wape": baseline.wape,
                **probability,
            }
        )

        frame = snapshot[["sku", "date", "target_end", "lead_time_demand"]].copy()
        frame["p10"] = forecast.p10
        frame["p50"] = forecast.p50
        frame["p90"] = forecast.p90
        frame["fold"] = fold.fold
        predictions.append(frame)

    metrics = pd.DataFrame(fold_records)
    all_predictions = pd.concat(predictions, ignore_index=True)
    if not (all_predictions["p10"] <= all_predictions["p50"]).all():
        raise ValueError("P10/P50 coherence failure.")
    if not (all_predictions["p50"] <= all_predictions["p90"]).all():
        raise ValueError("P50/P90 coherence failure.")
    return metrics, all_predictions
