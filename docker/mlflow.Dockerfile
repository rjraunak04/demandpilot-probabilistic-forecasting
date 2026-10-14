FROM python:3.12-slim

WORKDIR /app

# MLflow's lightweight package exposes the CLI but does not include the web
# server stack. Install the explicit server runtime dependencies while keeping
# forecasting/training libraries out of this image.
RUN pip install --no-cache-dir \
    "mlflow-skinny>=3,<4" \
    "flask>=3,<4" \
    "uvicorn>=0.32,<1"

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:////mlflow/mlflow.db", "--artifacts-destination", "/mlflow/artifacts"]
