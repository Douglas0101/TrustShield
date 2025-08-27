#!/usr/bin/env bash
set -euo pipefail

log(){ echo "[entrypoint] $(date -Is) $*"; }
log "Starting entrypoint.sh"

# Docker secrets -> env
if [ -z "${AWS_ACCESS_KEY_ID:-}" ] && [ -f /run/secrets/minio_root_user ]; then
  export AWS_ACCESS_KEY_ID="$(cat /run/secrets/minio_root_user)"
fi
if [ -z "${AWS_SECRET_ACCESS_KEY:-}" ] && [ -f /run/secrets/minio_root_password ]; then
  export AWS_SECRET_ACCESS_KEY="$(cat /run/secrets/minio_root_password)"
fi

# opcional: aguardar MLflow se for configurar MLFLOW_TRACKING_URI
: "${MLFLOW_TRACKING_URI:=}"
if [ -n "${MLFLOW_TRACKING_URI}" ]; then
  python - <<'PY' || true
import os, time, urllib.request
url = os.environ['MLFLOW_TRACKING_URI'].rstrip('/') + '/version'
for _ in range(180):
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            if 200 <= r.getcode() < 500: break
    except Exception: time.sleep(1)
PY
fi

log "Executing: $*"
exec "$@"
