"""MLflow tracking boundary."""

from contextlib import contextmanager
from pathlib import Path

import mlflow


def _tracking_uri(tracking_uri: str) -> str:
    if "://" in tracking_uri:
        return tracking_uri
    database = Path(tracking_uri).resolve()
    if database.suffix != ".db":
        database = database / "mlflow.db"
    database.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{database.as_posix()}"


def configure_mlflow(tracking_uri="mlruns", experiment_name="demandpilot"):
    mlflow.set_tracking_uri(_tracking_uri(tracking_uri))
    return mlflow.set_experiment(experiment_name).experiment_id


@contextmanager
def tracked_run(run_name, params=None, tags=None):
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        if params:
            mlflow.log_params(params)
        yield run.info.run_id


def log_metrics(metrics):
    mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
