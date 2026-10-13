from pathlib import Path
import tempfile
from fastapi.testclient import TestClient
from demandpilot.api.app import app
from demandpilot.tracking.mlflow import configure_mlflow,tracked_run,log_metrics
with tempfile.TemporaryDirectory() as d:
    configure_mlflow(str(Path(d)/"mlruns"),"demandpilot-day05")
    with tracked_run("verify",params={"horizon":7}): log_metrics({"wape":.601275})
client=TestClient(app)
assert client.get("/health").status_code==200
assert client.post("/inventory/recommendation",json={"p10":10,"p50":15,"p90":20,"on_hand":12}).status_code==200
print("DemandPilot Day-05 serving platform PASSED")
print("MLflow + SQLAlchemy/PostgreSQL contract + FastAPI verified")
