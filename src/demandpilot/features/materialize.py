"""Materialize validated DemandPilot forecasting datasets."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from demandpilot.features.daily import build_daily_demand
from demandpilot.features.pipeline import build_forecasting_features
from demandpilot.features.prepare import prepare_forecasting_transactions
from demandpilot.features.validate import validate_forecasting_features

DEFAULT_DAILY_PATH = Path(
    "data/processed/daily_sku_demand.parquet"
)

DEFAULT_FEATURE_PATH = Path(
    "data/processed/forecast_features.parquet"
)

DEFAULT_SUMMARY_PATH = Path(
    "reports/data_quality/feature_pipeline_summary.json"
)


def _write_parquet(
    data: pd.DataFrame,
    path: Path,
) -> None:
    """Write a dataframe to Parquet after creating its parent directory."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_parquet(
        path,
        index=False,
    )


def _write_json(
    payload: dict[str, Any],
    path: Path,
) -> None:
    """Write deterministic human-readable JSON."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def materialize_feature_pipeline(
    raw: pd.DataFrame,
    daily_path: Path = DEFAULT_DAILY_PATH,
    feature_path: Path = DEFAULT_FEATURE_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
) -> dict[str, Any]:
    """Build, validate, and persist Day-02 forecasting artifacts."""
    prepared, _ = prepare_forecasting_transactions(
        raw
    )

    daily, daily_report = build_daily_demand(
        prepared
    )

    features, pipeline_report = build_forecasting_features(
        raw
    )

    validate_forecasting_features(
        features
    )

    _write_parquet(
        daily,
        daily_path,
    )

    _write_parquet(
        features,
        feature_path,
    )

    summary: dict[str, Any] = {
        "pipeline": pipeline_report.as_dict(),
        "daily_demand": asdict(daily_report),
        "artifacts": {
            "daily_sku_demand": str(
                daily_path.as_posix()
            ),
            "forecast_features": str(
                feature_path.as_posix()
            ),
            "summary": str(
                summary_path.as_posix()
            ),
        },
        "feature_dataset": {
            "rows": int(len(features)),
            "columns": int(len(features.columns)),
            "date_min": features["date"].min().isoformat(),
            "date_max": features["date"].max().isoformat(),
            "unique_skus": int(
                features["sku"].nunique()
            ),
            "zero_demand_rows": int(
                features["demand"].eq(0).sum()
            ),
            "zero_demand_share": float(
                features["demand"].eq(0).mean()
            ),
        },
    }

    _write_json(
        summary,
        summary_path,
    )

    return summary