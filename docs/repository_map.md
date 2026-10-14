# Repository map

| Path | Responsibility |
| --- | --- |
| `src/demandpilot/data/` | ingestion, validation, schema, demand construction |
| `src/demandpilot/features/` | leakage-safe forecasting features |
| `src/demandpilot/models/` | baselines, training, tuning, probabilistic models |
| `src/demandpilot/inventory/` | uncertainty-aware inventory policy |
| `src/demandpilot/db/` | SQLAlchemy persistence contract |
| `src/demandpilot/api/` | FastAPI serving layer |
| `src/demandpilot/monitoring/` | PSI reports and drift alert policy |
| `dashboard/` | Streamlit decision interface |
| `scripts/` | reproducible operational/experiment entry points |
| `tests/` | automated data, ML, platform, API, and monitoring checks |
| `docker/` | runtime images |
| `.github/workflows/` | CI quality gates |
| `docs/` | architecture, model, deployment, and operational documentation |
| `data/` | local data zones; generated/private data excluded from Git |
| `reports/` | generated evaluation/monitoring outputs excluded where appropriate |

The repository keeps source code and contracts versioned while generated data,
model artifacts, MLflow state, reports, credentials, and local environments stay
outside source control.
