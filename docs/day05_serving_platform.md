# Day 05 — MLflow, PostgreSQL and FastAPI

Checkpoints: MLflow experiment tracking; metric/parameter logging; PostgreSQL-compatible SQLAlchemy forecast schema; repository persistence boundary; typed FastAPI contracts; forecast create/read and inventory endpoints; health/readiness behavior; integration verification.

Production persistence uses `DEMANDPILOT_DATABASE_URL`, for example `postgresql+psycopg://demandpilot:demandpilot@localhost:5432/demandpilot`. SQLite remains the zero-infrastructure test fallback.

MLflow uses a database-backed tracking store. A local path such as `mlruns` is normalized to `sqlite:///.../mlruns/mlflow.db`, avoiding MLflow's maintenance-mode filesystem tracking backend. A remote MLflow tracking URI can be supplied unchanged.

API routes are `GET /health`, `GET /ready`, `POST /forecasts`, `GET /forecasts/{sku}`, and `POST /inventory/recommendation`. OpenAPI is available at `/docs`.

Docker Compose and live PostgreSQL/MLflow services remain Day-06 infrastructure work.
