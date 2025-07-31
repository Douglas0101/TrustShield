# docker/mlflow.Dockerfile (Final e Verificado)
# Imagem personalizada para o serviço MLflow para incluir dependências.

# ATUALIZAÇÃO: Usa a versão estável mais recente e oficial do MLflow,
# verificada diretamente do repositório GitHub.
FROM ghcr.io/mlflow/mlflow:v3.1.4

# Instala as dependências necessárias para a conexão com PostgreSQL (backend) e MinIO (artifacts).
# Fazer isso no build da imagem é muito mais eficiente do que no runtime,
# resultando em um startup mais rápido e confiável do serviço.
RUN pip install boto3 psycopg2-binary