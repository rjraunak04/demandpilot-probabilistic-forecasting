from pathlib import Path

import pandas as pd

from demandpilot.evaluation.backtesting import backtest_baselines, summarize_backtest
from demandpilot.evaluation.split import chronological_split

FEATURE_PATH = Path("data/processed/forecast_features.parquet")
DETAIL_PATH = Path("reports/modeling/baseline_backtest_folds.csv")
SUMMARY_PATH = Path("reports/modeling/baseline_backtest_summary.csv")


def main() -> None:
    features = pd.read_parquet(FEATURE_PATH)
    split, _ = chronological_split(features)

    results = backtest_baselines(split.validation)
    summary = summarize_backtest(results)

    DETAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(DETAIL_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    assert len(results) == 8
    assert results["fold"].nunique() == 4
    assert summary["folds"].eq(4).all()
    assert results["validation_end"].max() == "2011-11-11"

    print("DemandPilot Day-03 CP-04 rolling-origin backtest PASSED")
    print("Protocol: 4 chronological folds x 7 days; locked test untouched")
    print()
    print(results.to_string(index=False))
    print()
    print("Backtest summary")
    print(summary.to_string(index=False))
    print()
    print(f"Saved: {DETAIL_PATH}")
    print(f"Saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
