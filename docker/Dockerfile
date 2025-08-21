# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependências primeiro para cache eficiente
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia o código e configs
COPY src/ src/
COPY config/ config/

# Entrypoint que espera o MLflow
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000
# o comando final é delegado ao compose (entrypoint.sh + uvicorn)
