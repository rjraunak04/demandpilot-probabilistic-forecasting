# Day 06 — Dashboard, containers and monitoring

Day 06 adds the operational surface around DemandPilot: a Streamlit forecast explorer, a live FastAPI serving service, PostgreSQL persistence, MLflow experiment infrastructure, Docker Compose orchestration, and a lightweight PSI-based drift signal.

## Compose services

- `postgres`: persistent PostgreSQL 17 database used by the serving layer.
- `mlflow`: persistent MLflow tracking server on port 5000 with a SQLite backend and artifact volume.
- `api`: FastAPI service on port 8000. It initializes the serving schema before Uvicorn starts and receives both the PostgreSQL URL and MLflow tracking URI through environment variables.
- `dashboard`: Streamlit operational UI on port 8501.

All long-running services have health checks and `restart: unless-stopped`. Startup dependencies use health conditions so the API does not start before PostgreSQL and MLflow are ready, and the dashboard waits for the API.

Local Compose credentials are configurable through `POSTGRES_DB`, `POSTGRES_USER`, and `POSTGRES_PASSWORD`. The checked-in values are development defaults only; production deployments must inject secrets through the deployment platform.

## Build hygiene

`.dockerignore` excludes the virtual environment, Git metadata, raw/processed data, artifacts, notebooks, reports, tests, caches, and local environment files from Docker build contexts. This keeps large private/local datasets out of container images and materially reduces build transfer size.

## Monitoring boundary

The current monitoring module implements deterministic PSI-based numeric feature drift with stable/watch/drift thresholds. It is intentionally lightweight and testable without external infrastructure. CP-04 adds JSON drift summaries, optional Evidently HTML, and a Streamlit drift tab. See [day06_monitoring.md](day06_monitoring.md).

## CP-05 — live decision dashboard

The dashboard now uses `DEMANDPILOT_API_URL` for live service health, stored forecast lookup, and inventory recommendations. The Inventory decision tab reads forecast quantiles from FastAPI/PostgreSQL and sends the operator-provided on-hand/MOQ values to the inventory policy endpoint. API failures are surfaced as UI errors rather than crashing Streamlit. The upload-based forecast explorer remains available for offline analysis.

## Local verification

Run `docker compose config`, then `docker compose up -d --build`. Verify all four services with `docker compose ps`, FastAPI `/health` and `/ready`, MLflow `/health`, PostgreSQL `pg_isready`, and Streamlit `/_stcore/health`.


## CP-06 — alerting and Day-06 acceptance

Monitoring is actionable rather than purely visual. PSI states map to stable, watch, and drift operator decisions, and `scripts/check_drift_status.py` exposes an exit-code contract for scheduled jobs. Material drift requests investigation before retraining; it never triggers blind automatic retraining.

Day-06 acceptance requires the full Python test suite, Ruff, byte-code compilation, clean Git state, and healthy Compose services. The verified CP-06 baseline contains 200 passing tests. CI/CD and external deployment automation are intentionally deferred to Day 07.
