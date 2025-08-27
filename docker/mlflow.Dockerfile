FROM ghcr.io/mlflow/mlflow:v3.3.1
RUN python -m pip install --no-cache-dir boto3 psycopg2-binary prometheus-client
