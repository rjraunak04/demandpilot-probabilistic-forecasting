# Day 05 — MLflow, PostgreSQL and FastAPI

Checkpoints: MLflow experiment tracking; metric/parameter logging; PostgreSQL-compatible SQLAlchemy forecast schema; repository persistence boundary; typed FastAPI contracts; forecast create/read and inventory endpoints; health/readiness behavior; integration verification.

Production database configuration uses `DEMANDPILOT_DATABASE_URL`, e.g. `postgresql+psycopg://demandpilot:demandpilot@localhost:5432/demandpilot`. SQLite remains a test fallback. API routes are `GET /health`, `GET /ready`, `POST /forecasts`, `GET /forecasts/{sku}`, and `POST /inventory/recommendation`. OpenAPI is available at `/docs`.

Docker Compose and live PostgreSQL/MLflow services remain Day-06 infrastructure work.
