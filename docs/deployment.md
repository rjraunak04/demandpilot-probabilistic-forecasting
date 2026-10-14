# DemandPilot deployment contract

DemandPilot is packaged as four cooperating services: PostgreSQL, MLflow,
FastAPI, and Streamlit. Docker Compose is the reproducible local integration
environment; a production platform may deploy the same responsibilities as
separate managed services.

## Required configuration

The application must receive database credentials from the deployment
environment. Copy `.env.example` only as a local template and replace
`POSTGRES_PASSWORD`; never commit a real password or platform secret.

The Compose network keeps PostgreSQL private. FastAPI, MLflow, and Streamlit
ports are exposed for local operation; production ingress should expose only
the endpoints required by the chosen architecture and should add TLS,
authentication, authorization, and platform-level secret management.

## Runtime hardening

The FastAPI and Streamlit images run as an unprivileged `demandpilot` user,
disable Python byte-code writes, and use unbuffered logs. Build context rules
exclude Git metadata, local environments, datasets, reports, tests, notebooks,
artifacts, and environment files.

Container health checks are the deployment contract:

- PostgreSQL: `pg_isready`
- MLflow: HTTP availability on port 5000
- FastAPI: `/health`, with `/ready` additionally checking database access
- Streamlit: `/_stcore/health`

## Release smoke test

After deployment, verify service health before routing production traffic:

```text
GET <api>/health
GET <api>/ready
GET <dashboard>/_stcore/health
```

A release must be rolled back or investigated when readiness fails. Model or
data drift is handled separately by the monitoring policy and never triggers
blind automatic deployment or retraining.

## Production boundary

The repository demonstrates deployable application contracts, but it does not
claim a public production deployment. Cloud-specific infrastructure,
authentication, TLS, backups, autoscaling, observability, and secret rotation
must be configured by the target platform before production use.
