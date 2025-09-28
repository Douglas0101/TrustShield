FROM ghcr.io/mlflow/mlflow:v3.3.1

ENV \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir prometheus-flask-exporter psycopg2-binary
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/* && pip install prometheus-flask-exporter psycopg2-binary boto3
