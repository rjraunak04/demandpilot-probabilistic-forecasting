"""Materialize Day-04 probabilistic forecasts and inventory intelligence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from demandpilot.evaluation.lead_time import build_lead_time_examples
from demandpilot.evaluation.probabilistic_backtest import probabilistic_backtest
from demandpilot.inventory.policy import inventory_recommendations

FEATURE_PATH = Path("data/processed/forecast_features.parquet")
REPORT_DIR = Path("reports/modeling")


def main() -> None:
    features = pd.read_parquet(FEATURE_PATH)
    examples = build_lead_time_examples(features, horizon_days=7)
    metrics, predictions = probabilistic_backtest(examples)

    decision_input = predictions[["sku", "date", "p10", "p50", "p90"]].copy()
    decisions = inventory_recommendations(decision_input)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORT_DIR / "probabilistic_backtest_metrics.csv"
    predictions_path = REPORT_DIR / "probabilistic_validation_forecasts.parquet"
    decisions_path = REPORT_DIR / "inventory_scenarios.parquet"
    summary_path = REPORT_DIR / "day04_summary.json"

    metrics.to_csv(metrics_path, index=False)
    predictions.to_parquet(predictions_path, index=False)
    decisions.to_parquet(decisions_path, index=False)

    summary = {
        "horizon_days": 7,
        "forecast_origins": metrics["origin"].tolist(),
        "forecast_rows": int(len(predictions)),
        "mean_p50_wape": float(metrics["p50_wape"].mean()),
        "mean_seasonal_7d_wape": float(metrics["seasonal_7d_wape"].mean()),
        "mean_coverage_80": float(metrics["coverage_80"].mean()),
        "mean_interval_width": float(metrics["mean_interval_width"].mean()),
        "inventory_policy": {
            "service_target": "P90 lead-time demand",
            "on_hand_available": False,
            "reorder_quantity_status": "requires real on-hand inventory",
        },
        "locked_test": "untouched",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("DemandPilot Day-04 probabilistic + inventory pipeline PASSED")
    print("Protocol: rolling weekly origins, horizon-safe 7-day lead-time target")
    print("Locked test: untouched")
    print()
    print(metrics.to_string(index=False))
    print()
    print(f"Forecast rows: {len(predictions):,}")
    print(f"Mean P50 WAPE: {summary['mean_p50_wape']:.6f}")
    print(f"Mean seasonal 7d WAPE: {summary['mean_seasonal_7d_wape']:.6f}")
    print(f"Mean 80% coverage: {summary['mean_coverage_80']:.2%}")
    print("Inventory quantities remain scenario-only until on-hand stock is supplied.")


if __name__ == "__main__":
    main()
