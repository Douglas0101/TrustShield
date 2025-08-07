#!/bin/sh
# ==============================================================================
# entrypoint.sh - TrustShield Enterprise Grade
# Versão: 9.0.0
#
# Otimizações e Melhores Práticas Implementadas:
# - Script 100% compatível com POSIX sh para máxima portabilidade.
# - Parametrização via variáveis de ambiente (WAIT_HOSTS, WAIT_TIMEOUT).
# - Loop de espera robusto com timeout para evitar bloqueios infinitos.
# - Logging claro e informativo.
# - Uso de 'exec' para passar o controle de processo corretamente.
# ==============================================================================

# Termina o script imediatamente se um comando falhar.
set -e

# --- Variáveis de Ambiente ---
# Exemplo: WAIT_HOSTS="postgres:5432,minio:9000"
# Se WAIT_HOSTS não for definido, o script continua sem esperar.
WAIT_HOSTS=${WAIT_HOSTS:-""}
WAIT_TIMEOUT=${WAIT_TIMEOUT:-120}

# --- Funções de Log ---
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ENTRYPOINT] INFO: $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ENTRYPOINT] ERROR: $1" >&2
    exit 1
}

# --- Função Principal de Espera ---
wait_for_services() {
    # Se a variável WAIT_HOSTS estiver vazia, não faz nada.
    if [ -z "$WAIT_HOSTS" ]; then
        log_info "Nenhum serviço para aguardar (WAIT_HOSTS não definido)."
        return
    fi

    # Transforma a string separada por vírgulas numa lista para o loop
    # IFS (Internal Field Separator) é alterado para a vírgula.
    IFS=','
    for service in $WAIT_HOSTS; do
        # Restaura o IFS para o padrão
        unset IFS

        # Separa host e porta
        host=$(echo "$service" | cut -d: -f1)
        port=$(echo "$service" | cut -d: -f2)

        log_info "Aguardando pelo serviço: ${host} na porta ${port}..."

        start_time=$(date +%s)
        # Usa 'nc' (netcat) para verificar se a porta está aberta.
        while ! nc -z "${host}" "${port}"; do
            sleep 2
            current_time=$(date +%s)
            elapsed_time=$((current_time - start_time))

            if [ ${elapsed_time} -ge "${WAIT_TIMEOUT}" ]; then
                log_error "Timeout! O serviço ${host}:${port} não ficou disponível em ${WAIT_TIMEOUT} segundos."
            fi
            # Imprime um ponto para feedback visual sem quebrar a linha.
            printf "."
        done
        # Adiciona uma nova linha para formatação limpa.
        printf "\\n"
        log_info "Serviço ${host}:${port} está pronto!"
        # Restaura o IFS para o próximo loop
        IFS=','
    done
    unset IFS
}

# --- Orquestração do Arranque ---
wait_for_services

log_info "Todos os serviços dependentes estão operacionais. Executando o comando principal..."

# 'exec' substitui o processo do shell pelo comando passado como argumento ($@).
# Esta é a melhor prática para gestão de sinais e processos em containers.
exec "$@"
