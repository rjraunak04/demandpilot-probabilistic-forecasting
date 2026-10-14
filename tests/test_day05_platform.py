from datetime import date
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from demandpilot.api.app import app
from demandpilot.db.models import Base, ForecastRecord
from demandpilot.db.repository import ForecastRepository
from demandpilot.tracking.mlflow import configure_mlflow, log_metrics, tracked_run


def test_mlflow(tmp_path: Path):
    configure_mlflow(
        str(tmp_path / "mlruns"),
        "demandpilot-test",
    )
    with tracked_run("test", params={"horizon": 7}) as run_id:
        log_metrics({"wape": 0.601275})
    assert run_id


def test_repository():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        repo = ForecastRepository(session)
        repo.add(
            ForecastRecord(
                sku="A",
                forecast_date=date(2011, 11, 12),
                p10=1,
                p50=2,
                p90=3,
            )
        )
        found = repo.get("A", date(2011, 11, 12))
        assert found is not None
        assert found.p50 == 2


def test_health_and_inventory():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    response = client.post(
        "/inventory/recommendation",
        json={"p10": 10, "p50": 15, "p90": 20, "on_hand": 12},
    )
    assert response.status_code == 200
    assert response.json()["reorder_quantity"] == 8
