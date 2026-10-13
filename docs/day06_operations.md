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

The current monitoring module implements deterministic PSI-based numeric feature drift with stable/watch/drift thresholds. It is intentionally lightweight and testable without external infrastructure. A richer report layer can be added independently without changing the serving contract.

## Local verification

Run `docker compose config`, then `docker compose up -d --build`. Verify all four services with `docker compose ps`, FastAPI `/health` and `/ready`, MLflow `/health`, PostgreSQL `pg_isready`, and Streamlit `/_stcore/health`.
