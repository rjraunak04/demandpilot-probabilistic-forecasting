FROM python:3.12-slim

WORKDIR /app

# The current dashboard is an operations UI over forecast exports; it does not
# need model-training libraries in its container.
RUN pip install --no-cache-dir \
    "pandas>=2.2,<3" \
    "pyarrow>=18,<26" \
    "streamlit>=1.50,<2"

COPY dashboard ./dashboard

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
