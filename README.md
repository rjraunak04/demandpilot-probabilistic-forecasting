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