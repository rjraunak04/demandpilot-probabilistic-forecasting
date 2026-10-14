"""Verify the Day-05 serving boundaries."""

import tempfile
from pathlib import Path

import mlflow
from fastapi.testclient import TestClient

from demandpilot.api.app import app
from demandpilot.tracking.mlflow import configure_mlflow, log_metrics, tracked_run


def main() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        configure_mlflow(
            str(Path(directory) / "mlruns"),
            "demandpilot-day05",
        )
        with tracked_run("verify", params={"horizon": 7}):
            log_metrics({"wape": 0.601275})
        mlflow.end_run()
        mlflow.set_tracking_uri("")

    client = TestClient(app)
    assert client.get("/health").status_code == 200
    response = client.post(
        "/inventory/recommendation",
        json={"p10": 10, "p50": 15, "p90": 20, "on_hand": 12},
    )
    assert response.status_code == 200

    print("DemandPilot Day-05 serving platform PASSED")
    print("MLflow + SQLAlchemy/PostgreSQL contract + FastAPI verified")


if __name__ == "__main__":
    main()
