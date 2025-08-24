#!/usr/bin/env bash
set -xeuo pipefail
echo "Starting entrypoint.sh"

# Secrets -> env (se ainda não vieram)
if [ -z "${AWS_ACCESS_KEY_ID:-}" ] && [ -f /run/secrets/minio_root_user ]; then
  export AWS_ACCESS_KEY_ID="$(cat /run/secrets/minio_root_user)"
fi
if [ -z "${AWS_SECRET_ACCESS_KEY:-}" ] && [ -f /run/secrets/minio_root_password ]; then
  export AWS_SECRET_ACCESS_KEY="$(cat /run/secrets/minio_root_password)"
fi

# Espera HTTP sem curl (usa Python da própria imagem)
wait_http_py() {
  local url="$1"; local tries="${2:-180}"
  python - <<PY
import sys, time, urllib.request
url = "$url"; tries = $tries
for i in range(tries):
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            if 200 <= r.getcode() < 500:
                sys.exit(0)
    except Exception:
        time.sleep(1)
sys.exit(1)
PY
}

: "${MLFLOW_TRACKING_URI:=http://mlflow:5000}"
wait_http_py "${MLFLOW_TRACKING_URI}/version" 180

echo "Executing command: $@"
exec "$@"
