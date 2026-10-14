FROM python:3.12-slim

WORKDIR /app

# Install the complete MLflow tracking server distribution, not mlflow-skinny.
# The full distribution resolves the server, SQL store, migration, and UI
# dependencies together. No DemandPilot training package is installed here.
RUN pip install --no-cache-dir "mlflow>=3,<4"

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--workers", "1", "--backend-store-uri", "sqlite:////mlflow/mlflow.db", "--artifacts-destination", "/mlflow/artifacts"]
