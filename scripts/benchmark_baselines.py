from pathlib import Path

import pandas as pd

from demandpilot.evaluation.benchmark import benchmark_baselines
from demandpilot.evaluation.split import chronological_split

FEATURE_PATH = Path("data/processed/forecast_features.parquet")
OUTPUT_PATH = Path("reports/modeling/baseline_validation_metrics.csv")


def main() -> None:
    features = pd.read_parquet(FEATURE_PATH)
    split, _ = chronological_split(features)
    validation = split.validation
    result = benchmark_baselines(validation)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    assert len(validation) == 81_092
    assert set(result["model"]) == {"naive_lag_1", "seasonal_naive_lag_7"}
    assert result["rows"].eq(81_092).all()

    print("DemandPilot Day-03 CP-03 baseline benchmark PASSED")
    print(f"Validation rows: {len(validation):,}")
    print()
    print(result.to_string(index=False))
    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
