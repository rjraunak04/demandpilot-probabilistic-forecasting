import pandas as pd

from demandpilot.evaluation.metrics import evaluate_forecast
from demandpilot.models.baselines import BASELINES


def benchmark_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Benchmark frame must not be empty.")
    if "demand" not in frame.columns:
        raise ValueError("Benchmark frame must contain demand.")

    records: list[dict[str, float | int | str]] = []
    for name, predictor in BASELINES.items():
        prediction = predictor(frame)
        metrics = evaluate_forecast(frame["demand"], prediction)
        records.append({"model": name, "rows": len(frame), **metrics.as_dict()})

    result = pd.DataFrame(records)
    return result.sort_values(["wape", "mae", "rmse", "model"]).reset_index(drop=True)
