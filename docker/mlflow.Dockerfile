# docker/mlflow.Dockerfile (Versão 2.0 - Corrigida com dependências de sistema)
# Imagem personalizada para o serviço MLflow para incluir dependências.

# Usa a versão estável e oficial do MLflow.
FROM ghcr.io/mlflow/mlflow:v3.1.4

# Instala as dependências de sistema E de Python.
# 1. netcat-openbsd: ESSENCIAL para o script de 'command' no docker-compose.yml funcionar.
# 2. boto3 e psycopg2-binary: Para conectar ao MinIO e PostgreSQL.
# Usamos --no-install-recommends para manter a imagem enxuta e limpamos o cache do apt.
RUN apt-get update && \
    apt-get install -y --no-install-recommends netcat-openbsd curl && \
    rm -rf /var/lib/apt/lists/* && \
    pip install boto3 psycopg2-binary