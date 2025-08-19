#!/usr/bin/env sh
set -eu

echo "[entrypoint] CONFIG_FILE=${CONFIG_FILE:-/app/config/config.yaml}"
echo "[entrypoint] DATA_DIR=${DATA_DIR:-/app/data} OUTPUTS_DIR=${OUTPUTS_DIR:-/app/outputs}"

wait_for() {
  host="$1"; port="$2"; name="$3"; timeout="${4:-120}"
  echo "[wait] Aguardando ${name} em ${host}:${port} (timeout: ${timeout}s)…"
  start="$(date +%s)"
  while :; do
    if nc -z "$host" "$port" >/dev/null 2>&1; then
      echo "[wait] ${name} pronto."
      break
    fi
    now="$(date +%s)"
    elapsed=$(( now - start ))
    if [ "$elapsed" -ge "$timeout" ]; then
      echo "[wait][ERRO] Timeout ao aguardar ${name} em ${host}:${port}"
      exit 1
    fi
    sleep 1
  done
}

wait_for postgres 5432 "Postgres"
wait_for minio    9000 "MinIO (S3)"
wait_for mlflow   5000 "MLflow Server"

[ -z "${BACKEND_STORE_URI:-}" ] && echo "[WARN] BACKEND_STORE_URI não definido."
[ -z "${MLFLOW_S3_ENDPOINT_URL:-}" ] && echo "[WARN] MLFLOW_S3_ENDPOINT_URL não definido."

exec "$@"
