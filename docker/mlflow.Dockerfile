FROM python:3.12-slim

WORKDIR /app

# Keep the tracking server intentionally small. Installing the DemandPilot
# package here would also pull forecasting/training dependencies that MLflow
# does not need at runtime.
RUN pip install --no-cache-dir "mlflow-skinny>=3,<4"

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:////mlflow/mlflow.db", "--default-artifact-root", "/mlflow/artifacts"]
