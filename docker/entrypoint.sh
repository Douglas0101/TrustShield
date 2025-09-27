#!/bin/bash
set -euo pipefail

# ============================================================================
# Entry point for the TrustShield containers
# ----------------------------------------------------------------------------
# 1. Reads credentials provided as Docker secrets (``*_FILE`` variables).
# 2. Ensures the project root and ``src`` directory are available in
#    ``PYTHONPATH`` for any subprocess executed by the container.
# 3. Finally, hands control to the command defined in the Dockerfile or the
#    ``docker-compose`` service definition.
# ============================================================================

read_secret() {
    local var_name="$1"
    local file_var_name="${var_name}_FILE"
    local file_path="${!file_var_name:-}"

    if [[ -n "${file_path}" && -f "${file_path}" ]]; then
        export "${var_name}"="$(<"${file_path}")"
    fi
}

read_secret "AWS_ACCESS_KEY_ID"
read_secret "AWS_SECRET_ACCESS_KEY"
read_secret "TRUSTSHIELD_API_KEYS"
read_secret "TRUSTSHIELD_ALLOWED_IPS"

if [[ -z "${TRUSTSHIELD_PROJECT_ROOT:-}" ]]; then
    export TRUSTSHIELD_PROJECT_ROOT="/app"
fi

case ":${PYTHONPATH:-}:" in
    *:"${TRUSTSHIELD_PROJECT_ROOT}/src":*) ;;
    *) export PYTHONPATH="${TRUSTSHIELD_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

exec "$@"
