import json
from pathlib import Path

import pandas as pd

from demandpilot.evaluation.split import chronological_split
from demandpilot.models.tuning import tune_model

FEATURE_PATH = Path("data/processed/forecast_features.parquet")
OUTPUT_PATH = Path("reports/modeling/optuna_tuning_results.json")
TRIALS = 8


def main() -> None:
    features = pd.read_parquet(FEATURE_PATH)
    split, _ = chronological_split(features)

    results = [
        tune_model(split.train, model=model, trials=TRIALS)
        for model in ("lightgbm", "xgboost")
    ]
    payload = {
        "selection_data": "training partition only",
        "locked_test_used": False,
        "trials_per_model": TRIALS,
        "results": [
            {
                "model": result.model,
                "best_wape": result.best_value,
                "best_params": result.best_params,
                "trials": result.trials,
            }
            for result in results
        ],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("DemandPilot Day-03 CP-06 time-aware Optuna tuning PASSED")
    print("Selection data: training partition only")
    print("Locked validation/test: untouched by tuning objective")
    for result in results:
        print(f"{result.model}: WAPE={result.best_value:.6f} params={result.best_params}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
