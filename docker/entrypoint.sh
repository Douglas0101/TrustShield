#!/bin/sh
# entrypoint.sh - TrustShield Project (Versão 7.0.1-aligned)
# Garante que serviços dependentes (MLflow, MinIO, PostgreSQL) estejam prontos antes de executar o comando principal.
# Alinhado com docker-compose.yml (v7.0.2-aligned) e Makefile (v7.0.0-aligned). Correções aplicadas para compatibilidade POSIX e linting.

set -e

# Variáveis configuráveis (podem ser sobrescritas via env no docker-compose)
WAIT_TIMEOUT=${WAIT_TIMEOUT:-60}  # Timeout total por serviço em segundos

# Função para verificar a saúde de um serviço com timeout
wait_for_service() {
  # Declarações separadas (correção para evitar masking de retornos)
  _host="$1"
  _port="$2"
  _service_name="$3"
  _timeout="$4"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: Aguardando pelo serviço: ${_service_name} em ${_host}:${_port}..."

  # Verifica se nc está disponível
  if ! command -v nc >/dev/null 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERRO: 'nc' não encontrado. Instale netcat-traditional no Dockerfile."
    exit 1
  fi

  # Declarações e atribuições separadas
  _start_time=""
  _start_time=$(date +%s)
  while ! nc -z "${_host}" "${_port}"; do
    sleep 2
    _current_time=""
    _current_time=$(date +%s)
    if [ $((_current_time - _start_time)) -ge "${_timeout}" ]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERRO: Timeout ao aguardar ${_service_name} após ${_timeout}s."
      exit 1
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: Aguardando..."
  done

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: ${_service_name} está pronto."
}

# Aguarda pelos serviços dependentes (com quoting em variáveis)
wait_for_service "postgres" "5432" "PostgreSQL" "${WAIT_TIMEOUT}"  # Backend para MLflow tracking
wait_for_service "minio" "9000" "MinIO" "${WAIT_TIMEOUT}"        # Storage para artefatos MLflow
wait_for_service "mlflow" "5000" "MLflow UI" "${WAIT_TIMEOUT}"   # Servidor MLflow

echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: Todos os serviços estão prontos. Executando o comando principal..."

# Executa o comando passado para o container (ex: python src/models/train_fraud_model.py)
exec "$@"
