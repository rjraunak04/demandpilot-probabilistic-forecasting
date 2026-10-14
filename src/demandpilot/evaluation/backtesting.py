from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from demandpilot.evaluation.benchmark import benchmark_baselines


@dataclass(frozen=True)
class BacktestFold:
    fold: int
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def rolling_origin_folds(
    *,
    validation_start: str = "2011-10-15",
    validation_end: str = "2011-11-11",
    horizon_days: int = 7,
) -> tuple[BacktestFold, ...]:
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")

    start = pd.Timestamp(validation_start)
    end = pd.Timestamp(validation_end)
    if start > end:
        raise ValueError("validation_start must not be after validation_end.")

    total_days = (end - start).days + 1
    if total_days % horizon_days != 0:
        raise ValueError("Validation window must contain complete forecast horizons.")

    folds: list[BacktestFold] = []
    fold_start = start
    fold_number = 1

    while fold_start <= end:
        fold_end = fold_start + timedelta(days=horizon_days - 1)
        folds.append(
            BacktestFold(
                fold=fold_number,
                train_end=fold_start - timedelta(days=1),
                validation_start=fold_start,
                validation_end=fold_end,
            )
        )
        fold_number += 1
        fold_start = fold_end + timedelta(days=1)

    return tuple(folds)


def backtest_baselines(
    frame: pd.DataFrame,
    *,
    validation_start: str = "2011-10-15",
    validation_end: str = "2011-11-11",
    horizon_days: int = 7,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Backtest frame must not be empty.")
    if "date" not in frame.columns:
        raise ValueError("Backtest frame must contain date.")

    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")

    records: list[dict[str, float | int | str]] = []
    for fold in rolling_origin_folds(
        validation_start=validation_start,
        validation_end=validation_end,
        horizon_days=horizon_days,
    ):
        validation = data.loc[
            data["date"].between(fold.validation_start, fold.validation_end)
        ].copy()
        if validation.empty:
            raise ValueError(f"Backtest fold {fold.fold} contains no rows.")

        metrics = benchmark_baselines(validation)
        for row in metrics.to_dict(orient="records"):
            records.append(
                {
                    "fold": fold.fold,
                    "train_end": str(fold.train_end.date()),
                    "validation_start": str(fold.validation_start.date()),
                    "validation_end": str(fold.validation_end.date()),
                    **row,
                }
            )

    return pd.DataFrame(records).sort_values(["fold", "wape", "model"]).reset_index(drop=True)


def summarize_backtest(results: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "wape", "mae", "rmse", "smape"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Missing backtest result columns: {sorted(missing)}")
    if results.empty:
        raise ValueError("Backtest results must not be empty.")

    summary = (
        results.groupby("model", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mean_wape=("wape", "mean"),
            std_wape=("wape", "std"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_smape=("smape", "mean"),
        )
        .sort_values(["mean_wape", "mean_mae", "model"])
        .reset_index(drop=True)
    )
    return summary
