FROM python:3.12-slim

WORKDIR /app

# Serving image: install only dependencies used by the API, database layer,
# and inventory policy. Training libraries stay out of the runtime image.
RUN pip install --no-cache-dir \
    "fastapi>=0.115,<1" \
    "numpy>=2.0,<3" \
    "pandas>=2.2,<3" \
    "psycopg[binary]>=3.2,<4" \
    "pydantic>=2.9,<3" \
    "sqlalchemy>=2.0,<3" \
    "uvicorn[standard]>=0.32,<1"

COPY src ./src
COPY scripts ./scripts

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "demandpilot.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
