# DemandPilot

**Production-Grade Probabilistic Retail Demand Forecasting & Inventory Intelligence Platform**

DemandPilot is an end-to-end machine learning system designed to transform
historical retail transactions into probabilistic demand forecasts and
inventory decision support.

The project is being developed as a production-oriented forecasting platform,
not only as a modeling notebook.

## Business Problem

Retail demand is uncertain and changes over time. Poor demand estimates can
lead to:

- stockouts and lost sales
- excess inventory
- unnecessary holding costs
- inefficient replenishment decisions

DemandPilot is designed to answer four core business questions:

1. What demand should be expected for each SKU?
2. Which products are at risk of stockout?
3. How much inventory should be reordered?
4. How uncertain is the demand forecast?

## System Vision

The planned end-to-end workflow is:

```text
Raw Retail Transactions
        |
        v
Data Ingestion
        |
        v
Data Validation & Quality Contracts
        |
        v
Daily SKU Demand Construction
        |
        v
Leakage-Safe Feature Engineering
        |
        v
Time-Series Backtesting
        |
        v
Baseline + Gradient Boosting Models
        |
        v
Probabilistic Forecasting
     P10 / P50 / P90
        |
        v
Inventory Intelligence
        |
        v
MLflow + PostgreSQL + FastAPI
        |
        v
Streamlit Dashboard
        |
        v
Monitoring + CI/CD + Deployment

## Forecasting Benchmark — Day 03

DemandPilot uses chronological evaluation rather than random train/test splits.

- Training ends: **2011-10-14**
- Validation: **2011-10-15 to 2011-11-11**
- Locked test: **2011-11-12 to 2011-12-09**
- Primary metric: **WAPE**
- Supporting metrics: MAE, RMSE, sMAPE
- Rolling-origin validation: four non-overlapping 7-day folds
- Models: last-value naive, seasonal naive, LightGBM, XGBoost
- Hyperparameter search: limited, time-aware Optuna tuning using training data only

The selected Day-03 champion is **LightGBM** with validation WAPE **0.860410**,
a **29.75%** improvement over the seasonal-naive WAPE of **1.224818**.
The locked test set remains untouched at model-selection time.

Generated benchmark CSV/JSON outputs under `reports/modeling/` are reproducible
runtime artifacts and are intentionally excluded from version control. The
evaluation logic, tests, configuration, and methodology remain versioned.


## Probabilistic Forecasting & Inventory Intelligence — Day 04

The operational forecasting target is a horizon-safe **7-day lead-time demand**
total. At each weekly forecast origin, features use only information available
before the origin and the target covers the following seven days.

Demand uncertainty is modeled with LightGBM quantile regression at **P10, P50,
and P90**. Evaluation includes pinball loss, median-forecast WAPE, empirical 80%
interval coverage, and interval width.

Inventory policy is deliberately auditable: P90 represents a service-oriented
target-stock scenario and P90-P50 is an uncertainty-derived safety-stock proxy.
Because Online Retail II does not provide a trustworthy current inventory
ledger, DemandPilot never fabricates on-hand stock or executable reorder
quantities. Those actions become available only when real on-hand inventory is
supplied.
