FROM ghcr.io/mlflow/mlflow:v3.3.1
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/* && pip install prometheus-flask-exporter psycopg2-binary