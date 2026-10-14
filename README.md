# DemandPilot

**Production-Grade Probabilistic Retail Demand Forecasting & Inventory Intelligence Platform**

DemandPilot is an end-to-end ML engineering project that turns historical
retail transactions into leakage-safe seven-day probabilistic SKU forecasts,
inventory decision support, production APIs, monitoring, and reproducible
containerized services.

## Why this project exists

Retail teams need more than a point forecast. They need to understand expected
demand, uncertainty, stockout exposure, and how a model behaves after
deployment. DemandPilot therefore treats forecasting as a complete ML system:
data contracts, chronological evaluation, uncertainty, serving, persistence,
monitoring, CI, and deployment boundaries are all first-class components.

## Verified result

The Day-04 validation benchmark contains **11,427 SKU-origin forecasts** across
four weekly origins. The selected probabilistic LightGBM model achieved **P50
WAPE 0.601275**, compared with **0.772276** for the seasonal seven-day baseline
(about **22.1% relative improvement**). The P10-P90 interval achieved **84.44%
empirical coverage**.

These are validation results. The locked test window remains outside model
selection and is not presented as production performance.

## Architecture

```text
UCI Online Retail II
        |
        v
Ingestion + data contracts
        |
        v
SKU x day demand panel
        |
        v
Leakage-safe lag / rolling features
        |
        v
Chronological backtesting
        |
        +--> naive + seasonal baselines
        +--> LightGBM / XGBoost
        |
        v
7-day LightGBM quantile forecasting
      P10 / P50 / P90
        |
        v
PostgreSQL forecast store
        |
   +----+------------------+
   |                       |
FastAPI              Inventory policy
   |                  P90 target stock
   v
Streamlit
        |
        v
PSI monitoring -> stable / watch / drift
        |
        v
investigate before retraining

MLflow: experiment tracking
Docker Compose: Postgres + MLflow + API + Dashboard
GitHub Actions: lint + tests + compile + container contract
```

Detailed design: `docs/architecture.md`.

## ML methodology

The operational target is total SKU demand over the next seven days. Features
at every forecast origin use only prior observations. Model selection is
chronological rather than randomly split.

DemandPilot evaluates simple baselines before gradient boosting. The final
probabilistic layer uses LightGBM quantile regression for P10, P50, and P90.
P50 is the central forecast; P10-P90 exposes uncertainty. P90 is used as a
service-oriented target-stock scenario only when real on-hand inventory is
provided.

The dataset has no trustworthy current inventory ledger, so the project never
fabricates on-hand stock or executable replenishment actions.

## Production and MLOps layer

- **FastAPI** — forecast persistence/retrieval and inventory recommendation API
- **PostgreSQL** — forecast persistence and readiness dependency
- **MLflow** — experiment tracking infrastructure
- **Streamlit** — forecast, inventory, and drift decision interface
- **Docker Compose** — reproducible four-service integration environment
- **PSI monitoring** — deterministic stable/watch/drift feature status
- **Drift gate** — explicit operational action; no blind auto-retraining
- **GitHub Actions** — Python 3.11/3.12 quality matrix, tests, compilation,
  Compose validation, and API/dashboard image builds
- **Non-root runtime images** — hardened API and dashboard containers

## Evaluation design

| Stage | Contract |
| --- | --- |
| Train end | 2011-10-14 |
| Validation | 2011-10-15 to 2011-11-11 |
| Locked test | 2011-11-12 to 2011-12-09 |
| Operational horizon | next 7 days |
| Primary point metric | WAPE |
| Uncertainty outputs | P10 / P50 / P90 |
| Interval metric | empirical P10-P90 coverage |

Earlier Day-03 daily-row modeling is retained as diagnostic model-development
evidence; Day-04's fixed-origin seven-day target is the operational forecasting
contract.

## Monitoring policy

Numeric drift uses reference-quantile PSI:

- **stable:** PSI < 0.10
- **watch:** 0.10 <= PSI < 0.25
- **drift:** PSI >= 0.25

A drift signal triggers investigation and fresh backtest review. It does not
automatically retrain or promote a replacement model.

## Run locally

Prerequisites: Docker Desktop with Compose.

```bash
git clone <repository-url>
cd demandpilot-probabilistic-forecasting
docker compose up -d --build
```

Local services:

- API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- MLflow: `http://localhost:5000`
- Dashboard: `http://localhost:8501`

For local credentials, copy `.env.example` and replace the example password.
Do not commit real credentials.

## Quality gates

```bash
python -m ruff check .
python -m pytest -q
python -m compileall -q src scripts dashboard
docker compose config --quiet
```

The accepted Day-07 baseline contains **200 passing tests** with one known
non-blocking SQLAlchemy/MLflow deprecation warning. CI independently executes
the repository quality and container contract on supported merge targets.

## Repository guide

| Area | Purpose |
| --- | --- |
| `src/demandpilot/data` | ingestion, validation, daily demand |
| `src/demandpilot/features` | leakage-safe feature engineering |
| `src/demandpilot/models` | baselines, training, probabilistic models |
| `src/demandpilot/inventory` | uncertainty-aware inventory policy |
| `src/demandpilot/db` | persistence layer |
| `src/demandpilot/api` | serving API |
| `src/demandpilot/monitoring` | drift reports and alert policy |
| `dashboard` | Streamlit interface |
| `tests` | automated quality contracts |
| `docs` | architecture, model card, deployment, operations |

See `docs/repository_map.md` for the full map.

## Engineering documentation

- `docs/architecture.md` — end-to-end architecture and boundaries
- `docs/model_card.md` — intended use, evidence, risks, and governance
- `docs/deployment.md` — runtime/deployment contract
- `docs/day06_monitoring.md` — drift methodology
- `docs/day06_operations.md` — monitoring operations
- `docs/day07_cicd.md` — CI/CD and release hardening
- `docs/release_checklist.md` — final acceptance criteria

## Limitations

This repository is a production-oriented engineering demonstration, not a claim
of a live commercial deployment. The source dataset is historical. A real
production rollout would additionally require retailer-specific features,
current inventory and lead-time data, authentication/authorization, TLS,
managed secrets, backups, autoscaling, platform observability, and business
approval workflows.

## Project status

**Days 01-07 implemented.** The repository now covers data engineering,
leakage-safe forecasting, probabilistic uncertainty, inventory intelligence,
experiment tracking, persistence, API serving, dashboarding, drift monitoring,
CI, and deployment hardening. Final release acceptance is documented in
`docs/release_checklist.md`.
