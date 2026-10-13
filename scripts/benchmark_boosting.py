from pathlib import Path

import pandas as pd

from demandpilot.evaluation.benchmark import benchmark_baselines
from demandpilot.evaluation.metrics import evaluate_forecast
from demandpilot.evaluation.split import chronological_split
from demandpilot.models.boosting import fit_predict, model_specs

FEATURE_PATH = Path("data/processed/forecast_features.parquet")
OUTPUT_PATH = Path("reports/modeling/boosting_validation_metrics.csv")


def main() -> None:
    features = pd.read_parquet(FEATURE_PATH)
    split, _ = chronological_split(features)
    train = split.train
    validation = split.validation

    records = benchmark_baselines(validation).to_dict(orient="records")
    for spec in model_specs():
        prediction = fit_predict(spec.estimator, train, validation)
        metrics = evaluate_forecast(validation["demand"], prediction)
        records.append({"model": spec.name, "rows": len(validation), **metrics.as_dict()})

    result = (
        pd.DataFrame(records)
        .sort_values(["wape", "mae", "rmse", "model"])
        .reset_index(drop=True)
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    assert len(train) == 2_018_615
    assert len(validation) == 81_092
    assert {"lightgbm", "xgboost", "naive_lag_1", "seasonal_naive_lag_7"} == set(
        result["model"]
    )

    print("DemandPilot Day-03 CP-05 boosting benchmark PASSED")
    print(f"Training rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print("Locked test: untouched")
    print()
    print(result.to_string(index=False))
    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
