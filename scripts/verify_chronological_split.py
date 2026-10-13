import pandas as pd

from demandpilot.evaluation.split import chronological_split

FEATURE_PATH = "data/processed/forecast_features.parquet"


def main() -> None:
    features = pd.read_parquet(FEATURE_PATH)

    split, report = chronological_split(features)

    assert split.train["date"].max() < split.validation["date"].min()
    assert split.validation["date"].max() < split.test["date"].min()
    assert split.validation["date"].nunique() == 28
    assert split.test["date"].nunique() == 28

    print("DemandPilot Day-03 CP-01 chronological split PASSED")
    print(f"Input rows: {report.input_rows:,}")
    print(f"Eligible rows: {report.eligible_rows:,}")
    print(
        "Excluded incomplete-history rows: "
        f"{report.excluded_incomplete_history_rows:,}"
    )
    print(f"Train rows: {report.train_rows:,}")
    print(f"Validation rows: {report.validation_rows:,}")
    print(f"Test rows: {report.test_rows:,}")
    print(f"Train: {report.train_start} -> {report.train_end}")
    print(
        f"Validation: {report.validation_start} -> "
        f"{report.validation_end}"
    )
    print(f"Locked test: {report.test_start} -> {report.test_end}")


if __name__ == "__main__":
    main()