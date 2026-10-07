"""Verify DemandPilot data contracts against the real retail dataset."""

from demandpilot.data.ingest import load_retail_data
from demandpilot.data.validate import validate_retail_data


def main() -> None:
    """Run raw-data contracts against Online Retail II."""
    data = load_retail_data()
    report = validate_retail_data(data)

    assert report.passed
    assert report.rows == len(data)
    assert report.columns == data.shape[1]

    print("DemandPilot CP-04 data contracts PASSED")
    print(f"Validated rows: {report.rows:,}")
    print(f"Validated columns: {report.columns}")
    print(f"Checks passed: {sum(report.checks.values())}/{len(report.checks)}")

    print("\nObserved business/data-quality conditions:")
    for name, value in report.observations.items():
        print(f"- {name}: {value:,}")


if __name__ == "__main__":
    main()