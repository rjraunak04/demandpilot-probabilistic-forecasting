FROM python:3.12-slim

WORKDIR /app

# Keep MLflow tracking-server runtime explicit and lightweight. mlflow-skinny
# provides the CLI/core client, while Flask/Uvicorn/SQLAlchemy supply the
# server and SQLite-backed tracking-store dependencies. Training libraries
# intentionally stay out of this image.
RUN pip install --no-cache-dir \
    "mlflow-skinny>=3,<4" \
    "flask>=3,<4" \
    "sqlalchemy>=2,<3" \
    "uvicorn>=0.32,<1"

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:////mlflow/mlflow.db", "--artifacts-destination", "/mlflow/artifacts"]
