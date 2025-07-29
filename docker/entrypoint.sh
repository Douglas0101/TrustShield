#!/bin/sh
# entrypoint.sh - TrustShield Project (Versão 8.1.0-posix-compliant)
# Garante que os serviços dependentes estejam totalmente operacionais antes de executar o comando principal.
# Script com sintaxe corrigida para ser 100% compatível com POSIX sh.

# Termina o script imediatamente se um comando falhar.
set -e

# --- Variáveis de Ambiente ---
WAIT_TIMEOUT=${WAIT_TIMEOUT:-90}

# --- Função de Log ---
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ENTRYPOINT] INFO: $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ENTRYPOINT] ERROR: $1" >&2
    exit 1
}

# --- Função para Aguardar por Serviços ---
wait_for_service() {
    # CORREÇÃO: Removido 'local', que não é POSIX. Em 'sh', variáveis
    # dentro de funções já têm escopo local por padrão.
    host="$1"
    port="$2"
    service_name="$3"
    timeout="$4"

    log_info "Aguardando pelo serviço: ${service_name} em ${host}:${port}..."

    # CORREÇÃO: Declaração e atribuição separadas para evitar mascarar o código de saída do comando.
    start_time=""
    start_time=$(date +%s)

    # CORREÇÃO: Utiliza 'printf' em vez de 'echo -n', que é o padrão POSIX
    # para imprimir sem uma nova linha.
    while ! nc -z "${host}" "${port}"; do
        sleep 2

        current_time=""
        current_time=$(date +%s)

        # CORREÇÃO: Variáveis dentro de expansão aritmética devem ser usadas sem '$'.
        if [ $((current_time - start_time)) -ge "${timeout}" ]; then
            log_error "Timeout ao aguardar por ${service_name} após ${timeout} segundos."
        fi
        printf "."
    done

    # Adiciona uma nova linha para uma formatação de log mais limpa após os pontos.
    printf "\n"
    log_info "Serviço ${service_name} está pronto!"
}

# --- Orquestração do Arranque ---
# CORREÇÃO: Garante que a variável de ambiente está entre aspas para
# prevenir "word splitting" caso ela contenha espaços.
wait_for_service "postgres" "5432" "PostgreSQL" "${WAIT_TIMEOUT}"
wait_for_service "minio" "9000" "MinIO" "${WAIT_TIMEOUT}"
wait_for_service "mlflow" "5000" "MLflow UI" "${WAIT_TIMEOUT}"

log_info "Todos os serviços estão operacionais. Executando o comando principal..."

# 'exec' substitui o processo do shell pelo comando, o que é uma boa prática.
exec "$@"