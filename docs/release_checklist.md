# Release acceptance checklist

A DemandPilot release candidate is accepted only when all applicable checks
below pass.

## Code quality

- `python -m ruff check .`
- `python -m pytest -q`
- `python -m compileall -q src scripts dashboard`
- `git diff --check`
- clean Git working tree

## Container contract

- `docker compose config --quiet`
- API and dashboard images build successfully
- PostgreSQL, MLflow, API, and dashboard report healthy
- API `/health` returns ok
- API `/ready` returns ready
- Streamlit `/_stcore/health` returns ok
- API and dashboard run as an unprivileged user

## ML governance

- model-selection evidence is validation-only unless locked-test evaluation is
  explicitly recorded
- no future information enters origin features
- forecast quantiles remain ordered
- inventory decisions do not fabricate on-hand inventory
- drift never causes blind automatic model promotion

## Repository hygiene

- no raw/private dataset committed
- no secrets or `.env` committed
- generated artifacts are excluded
- architecture, model card, deployment contract, and README match the code
- release commit is reproducible from the documented environment

A Git tag/release should be created only after this checklist and CI acceptance.
