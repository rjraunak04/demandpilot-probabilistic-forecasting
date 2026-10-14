FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "pandas>=2.2,<3" \
    "pyarrow>=18,<26" \
    "streamlit>=1.50,<2"

COPY dashboard ./dashboard
COPY src/demandpilot/monitoring ./src/demandpilot/monitoring

ENV PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN addgroup --system demandpilot \
    && adduser --system --ingroup demandpilot demandpilot \
    && chown -R demandpilot:demandpilot /app

USER demandpilot

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
