FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY dashboard ./dashboard
EXPOSE 8501
CMD ["streamlit","run","dashboard/app.py","--server.address=0.0.0.0"]
