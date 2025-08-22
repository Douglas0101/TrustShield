# docker/mlflow.Dockerfile
# Base oficial do MLflow (linha 3.x). Se precisar fixar, altere a tag abaixo.
FROM ghcr.io/mlflow/mlflow:v3.3.1

# Adiciona clientes necessários para Postgres + S3/MinIO
RUN pip install --no-cache-dir boto3 psycopg2-binary
