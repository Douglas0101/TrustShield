# ==============================================================================
# mlflow.Dockerfile - TrustShield Enterprise Grade
# Versão: 4.0.0
#
# Otimizações e Melhores Práticas Implementadas:
# - ATUALIZAÇÃO: Uso da imagem oficial e versionada do MLflow para estabilidade.
# - Pré-instalação de dependências para um startup de serviço mais rápido.
# ==============================================================================

# SEGURANÇA E ESTABILIDADE: Use uma tag de versão específica da imagem oficial.
# ATUALIZADO para a versão mais recente conforme solicitado.
FROM ghcr.io/mlflow/mlflow:v3.2.0

# Instala as dependências necessárias para a conexão com:
# - PostgreSQL (usado como backend-store)
# - MinIO/S3 (usado como artifact-store)
# Fazer isso no build da imagem é a prática recomendada para performance e confiabilidade.
RUN pip install --no-cache-dir boto3 psycopg2-binary
