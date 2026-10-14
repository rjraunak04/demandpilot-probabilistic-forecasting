# Day-06 CP-04: Monitoring and drift reports

Reference and current snapshots are compared using **numeric PSI** with reference
quantile bins. PSI < 0.10 = stable; 0.10–<0.25 = watch; >=0.25 = drift.
These are investigation heuristics, not evidence of reduced forecast accuracy.

## Commands

```powershell
python scripts/generate_drift_report.py --reference reference.csv --current current.csv --output reports/monitoring/drift.json
python -m pip install -e ".[monitoring]"
python scripts/generate_drift_report.py --reference reference.csv --current current.csv --evidently-html reports/monitoring/evidently.html
```

CSV and Parquet are supported. Use genuinely time-separated datasets with
matching feature definitions. No data or monitoring results are fabricated.

The Streamlit Drift monitoring tab compares uploaded datasets and downloads
JSON. Evidently is an **optional** HTML adapter and is deliberately excluded
from serving Docker images. Drift alerts should trigger data quality checks
and labeled backtest review, not automatic model retraining or replenishment.
