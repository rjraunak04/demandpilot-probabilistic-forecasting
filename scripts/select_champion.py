import json
from pathlib import Path

import pandas as pd

from demandpilot.evaluation.benchmark import benchmark_baselines
from demandpilot.evaluation.split import chronological_split
from demandpilot.models.selection import evaluate_tuned_models, select_champion

FEATURE_PATH = Path("data/processed/forecast_features.parquet")
METRICS_PATH = Path("reports/modeling/model_selection_validation.csv")
SELECTION_PATH = Path("reports/modeling/champion_model.json")


def main() -> None:
    features = pd.read_parquet(FEATURE_PATH)
    split, _ = chronological_split(features)

    baseline = benchmark_baselines(split.validation)
    seasonal = baseline.loc[baseline["model"] == "seasonal_naive_lag_7"].iloc[0]
    tuned = evaluate_tuned_models(split.train, split.validation)
    selection = select_champion(tuned, baseline_wape=float(seasonal["wape"]))

    comparison = pd.concat([tuned, baseline], ignore_index=True).sort_values(
        ["wape", "mae", "model"]
    )
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(METRICS_PATH, index=False)
    SELECTION_PATH.write_text(
        json.dumps(
            {
                "champion": selection.champion,
                "primary_metric": selection.primary_metric,
                "validation_wape": selection.validation_wape,
                "seasonal_baseline_wape": selection.baseline_wape,
                "improvement_over_seasonal_baseline_pct": (
                    selection.improvement_over_baseline_pct
                ),
                "params": selection.params,
                "locked_test_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert selection.validation_wape < selection.baseline_wape
    assert len(split.test) == 70_360

    print("DemandPilot Day-03 CP-07 model selection PASSED")
    print("Primary metric: WAPE")
    print("Locked test: untouched")
    print()
    print(comparison.to_string(index=False))
    print()
    print(f"Champion: {selection.champion}")
    print(f"Validation WAPE: {selection.validation_wape:.6f}")
    print(
        "Improvement vs seasonal naive: "
        f"{selection.improvement_over_baseline_pct:.2f}%"
    )
    print(f"Saved: {METRICS_PATH}")
    print(f"Saved: {SELECTION_PATH}")


if __name__ == "__main__":
    main()
