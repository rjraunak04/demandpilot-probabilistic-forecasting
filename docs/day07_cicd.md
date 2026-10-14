# Day 07 — CI/CD and release hardening

## CP-01: continuous integration

DemandPilot runs an automated quality gate on pushes and pull requests targeting
`main` or `develop`, with an explicit manual trigger for release checks.

The Python matrix covers 3.11 and 3.12, matching the package contract. Each
matrix job installs the project, runs Ruff, executes the complete pytest suite,
and byte-compiles production Python entry points.

A separate Docker job validates the Compose contract and builds the FastAPI and
Streamlit serving images. PostgreSQL uses the official image and MLflow is
excluded from routine CI image builds because its full image is comparatively
heavy; MLflow behavior remains covered by the Python platform tests and local
Day-06 Compose acceptance.

CI never requires the private/raw Online Retail II dataset. Generated data,
model artifacts, reports, local environment files, and credentials remain
outside version control.

## Merge gate

A release candidate is acceptable only when:

1. both Python matrix jobs pass;
2. Docker Compose configuration validates;
3. API and dashboard images build;
4. no generated artifacts or secrets are committed.

Repository branch protection can require the CI checks before merging to
`main`.

## CP-02: deployment and runtime hardening

The API and dashboard runtime images now execute as an unprivileged `demandpilot`
user and use unbuffered Python logging. `.env.example` documents local database
configuration without storing real credentials. The deployment contract in
`docs/deployment.md` defines health/readiness smoke tests and clearly separates
the repository's deployable container contract from cloud-specific production
claims.

CP-02 intentionally does not pretend that a public cloud deployment exists.
TLS, authentication, backups, autoscaling, platform observability, and secret
rotation belong to the selected deployment environment.

## CP-03: architecture documentation

`docs/architecture.md` records the offline ML, online serving, inventory, and
operations boundaries plus the end-to-end system flow.

## CP-04: model governance

`docs/model_card.md` records intended use, historical-data limitations,
leakage-safe target design, verified validation evidence, uncertainty semantics,
risks, and human-review requirements.

## CP-05: repository map

`docs/repository_map.md` maps source, tests, runtime, operations, and generated
artifact boundaries so reviewers can navigate the project quickly.

## CP-06: recruiter-facing README

The root README is the concise project entry point: business problem, verified
metrics, architecture, methodology, production stack, monitoring policy,
quick-start instructions, limitations, and links to engineering evidence.

## CP-07: release governance

`docs/release_checklist.md` defines code, container, ML-governance, and
repository-hygiene acceptance criteria. A release/tag is intentionally deferred
until the final checks and CI acceptance are confirmed.

## CP-08: final acceptance gate

Day 07 closes only after the documented quality commands, runtime health checks,
non-root checks, and clean Git state pass on the release candidate. This avoids
claiming completion from documentation alone.
