"""MLflow tracking boundary."""
from contextlib import contextmanager
from pathlib import Path
import mlflow

def configure_mlflow(tracking_uri="mlruns", experiment_name="demandpilot"):
    uri = tracking_uri if "://" in tracking_uri else Path(tracking_uri).resolve().as_uri()
    mlflow.set_tracking_uri(uri)
    return mlflow.set_experiment(experiment_name).experiment_id

@contextmanager
def tracked_run(run_name, params=None, tags=None):
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        if params: mlflow.log_params(params)
        yield run.info.run_id

def log_metrics(metrics):
    mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
