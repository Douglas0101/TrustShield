# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# curl para healthchecks e utilidades
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install "mlflow" boto3 psycopg2-binary

EXPOSE 5000
# o comando real vem do compose (mlflow server ...)
CMD ["mlflow","server","--host","0.0.0.0","--port","5000"]

