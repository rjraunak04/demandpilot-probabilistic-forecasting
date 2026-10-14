"""Materialize DemandPilot Day-02 forecasting artifacts."""

from demandpilot.data.ingest import load_retail_data
from demandpilot.features.materialize import (
    materialize_feature_pipeline,
)


def main() -> None:
    """Build and persist validated forecasting datasets."""
    raw = load_retail_data()

    summary = materialize_feature_pipeline(
        raw
    )

    pipeline = summary["pipeline"]
    dataset = summary["feature_dataset"]

    print(
        "DemandPilot Day-02 feature materialization PASSED"
    )
    print(
        f"Raw rows: {pipeline['raw_rows']:,}"
    )
    print(
        f"Prepared rows: {pipeline['prepared_rows']:,}"
    )
    print(
        "Observed SKU-day rows: "
        f"{pipeline['observed_daily_rows']:,}"
    )
    print(
        "Continuous panel rows: "
        f"{pipeline['panel_rows']:,}"
    )
    print(
        "Inserted zero-demand rows: "
        f"{pipeline['inserted_zero_rows']:,}"
    )
    print(
        f"Feature rows: {dataset['rows']:,}"
    )
    print(
        f"Feature columns: {dataset['columns']:,}"
    )
    print(
        f"Unique SKUs: {dataset['unique_skus']:,}"
    )
    print(
        "Complete-history share: "
        f"{pipeline['complete_history_share']:.2%}"
    )
    print(
        "Zero-demand share: "
        f"{dataset['zero_demand_share']:.2%}"
    )

    print(
        "\nArtifacts:"
    )

    for name, path in summary["artifacts"].items():
        print(
            f"- {name}: {path}"
        )


if __name__ == "__main__":
    main()