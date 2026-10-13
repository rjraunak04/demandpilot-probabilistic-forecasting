# Day 06 — Dashboard, containers and monitoring

Day 06 adds the operational surface around DemandPilot: a Streamlit forecast explorer, container images for the API and dashboard, PostgreSQL orchestration through Docker Compose, and a lightweight PSI-based drift signal that can be tested without external infrastructure.

The dashboard consumes forecast exports and surfaces SKU-level P10/P50/P90 uncertainty. The Compose stack starts PostgreSQL first, initializes the serving schema, then exposes FastAPI on port 8000 and Streamlit on port 8501. Docker availability is an environment prerequisite; unit tests do not require Docker.
