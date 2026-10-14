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


## CP-06 — operational alert policy

Drift summaries now map to explicit operator actions instead of being passive metrics only:

- **stable** -> `ok / continue_monitoring`
- **watch** -> `warning / monitor_next_window`
- **drift** -> `critical / investigate_before_retraining`

The policy deliberately does not auto-retrain a model when drift appears. Drift is a diagnostic signal, not proof that a replacement model is better. The Streamlit monitoring tab surfaces the action, while `scripts/check_drift_status.py` provides a scheduler/CI-friendly exit-code gate. By default it exits with code 2 only for material drift; `--fail-on watch` makes the gate stricter and `--fail-on never` keeps it informational.
