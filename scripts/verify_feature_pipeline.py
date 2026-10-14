"""Verify the complete DemandPilot Day-02 feature pipeline."""

from demandpilot.data.ingest import load_retail_data
from demandpilot.features.pipeline import build_forecasting_features
from demandpilot.features.validate import (
    expected_feature_columns,
    validate_forecasting_features,
)


def main() -> None:
    """Run end-to-end feature engineering on the complete dataset."""
    raw = load_retail_data()

    features, report = build_forecasting_features(
        raw
    )

    validate_forecasting_features(features)

    required = expected_feature_columns()

    assert required.issubset(features.columns)
    assert len(features) == report.feature_rows

    assert (
        report.complete_history_rows
        + report.incomplete_history_rows
        == report.feature_rows
    )

    assert (
        report.inserted_zero_rows
        == report.panel_rows
        - report.observed_daily_rows
    )

    print(
        "DemandPilot CP-06 feature pipeline validation PASSED"
    )
    print(
        f"Raw rows: {report.raw_rows:,}"
    )
    print(
        f"Prepared rows: {report.prepared_rows:,}"
    )
    print(
        "Observed SKU-day rows: "
        f"{report.observed_daily_rows:,}"
    )
    print(
        "Continuous panel rows: "
        f"{report.panel_rows:,}"
    )
    print(
        "Inserted zero-demand rows: "
        f"{report.inserted_zero_rows:,}"
    )
    print(
        "Final feature rows: "
        f"{report.feature_rows:,}"
    )
    print(
        f"Unique SKUs: {report.unique_skus:,}"
    )
    print(
        "Complete-history rows: "
        f"{report.complete_history_rows:,}"
    )
    print(
        "Incomplete-history rows: "
        f"{report.incomplete_history_rows:,}"
    )
    print(
        "Complete-history share: "
        f"{report.complete_history_share:.2%}"
    )
    print(
        "Required feature columns: "
        f"{len(required)}"
    )


if __name__ == "__main__":
    main()