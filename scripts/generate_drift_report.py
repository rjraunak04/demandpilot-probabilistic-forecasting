"""Generate monitoring artifacts from CSV or Parquet snapshots."""
import argparse
from pathlib import Path

import pandas as pd

from demandpilot.monitoring.report import save_evidently_html, save_summary, summarize_drift


def load(path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("input must be CSV or Parquet")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--columns", nargs="+")
    parser.add_argument("--output", type=Path, default=Path("reports/monitoring/drift.json"))
    parser.add_argument("--evidently-html", type=Path)
    args = parser.parse_args()
    reference, current = load(args.reference), load(args.current)
    report = summarize_drift(reference, current, columns=args.columns)
    print(f"PSI report: {save_summary(report, args.output)}")
    print(f"Status: {report['overall_status']}")
    if args.evidently_html:
        print(f"Evidently HTML: {save_evidently_html(reference, current, args.evidently_html)}")


if __name__ == "__main__":
    main()
