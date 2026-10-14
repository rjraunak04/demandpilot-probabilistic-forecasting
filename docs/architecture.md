# DemandPilot system architecture

## Objective

DemandPilot converts retail transactions into auditable seven-day probabilistic
SKU demand forecasts and inventory decision support. The system separates
offline model development from online serving so model selection cannot leak
future information into evaluation.

## End-to-end flow

```text
Online Retail II
      |
      v
ingestion + data contracts
      |
      v
SKU x day demand panel
      |
      v
lag / rolling features (past only)
      |
      v
chronological backtesting
      |
      +--> naive / seasonal baselines
      +--> LightGBM / XGBoost
      |
      v
LightGBM quantile models
 P10      P50      P90
      |
      v
forecast persistence (PostgreSQL)
      |
      +--> FastAPI --> Streamlit
      |
      +--> inventory policy (P90 target stock)
      |
      v
PSI drift monitoring + explicit alert gate
```

MLflow provides experiment tracking. Docker Compose integrates PostgreSQL,
MLflow, FastAPI, and Streamlit. GitHub Actions runs lint, tests, compilation,
Compose validation, and serving-image builds.

## ML boundary

The operational target is total demand over the next seven days. Features at a
forecast origin use only information available before that origin. Validation
uses chronological origins rather than random splitting. P10/P50/P90 quantile
models expose uncertainty instead of presenting a single point estimate as
certain.

The locked test period is deliberately excluded from model selection. This
repository reports validation results until the explicit final-test evaluation
is performed.

## Serving boundary

Forecast generation/training is an offline concern. The API persists and serves
forecast records and converts supplied forecast quantiles plus real on-hand
inventory into an inventory recommendation. Online Retail II has no trustworthy
current inventory ledger, so DemandPilot never invents on-hand stock.

## Operations boundary

PSI monitoring classifies features as stable, watch, or drift. Drift creates an
investigation/retraining-review action; it does not automatically replace the
model. Readiness checks cover database connectivity, while container health
checks cover each runtime service.

See `docs/deployment.md` for the deployment contract.
