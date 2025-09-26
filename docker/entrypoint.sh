#!/bin/bash
set -e

# ====================================================================
# SCRIPT DE ENTRADA INTELIGENTE PARA O TRUSTSHIELD API
#
# Função: Resolver segredos do Docker para variáveis de ambiente
# que a aplicação (boto3/mlflow) espera.
# ====================================================================

# Se a variável AWS_ACCESS_KEY_ID_FILE existir, leia o segredo do arquivo
# e exporte-o para a variável AWS_ACCESS_KEY_ID.
if [ -n "$AWS_ACCESS_KEY_ID_FILE" ]; then
    export AWS_ACCESS_KEY_ID=$(cat "$AWS_ACCESS_KEY_ID_FILE")
fi

# Faça o mesmo para a Secret Key.
if [ -n "$AWS_SECRET_ACCESS_KEY_FILE" ]; then
    export AWS_SECRET_ACCESS_KEY=$(cat "$AWS_SECRET_ACCESS_KEY_FILE")
fi

# Garante que os serviços internos conheçam a raiz do projeto.
if [ -z "$TRUSTSHIELD_PROJECT_ROOT" ]; then
    export TRUSTSHIELD_PROJECT_ROOT="/app"
fi

# Mantém o diretório ``src`` disponível no PYTHONPATH, mesmo quando o
# repositório é movido para outro caminho no host.
case ":$PYTHONPATH:" in
    *:"$TRUSTSHIELD_PROJECT_ROOT/src":*) ;;
    *) export PYTHONPATH="$TRUSTSHIELD_PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

# Agora, execute o comando principal que foi passado para o container
# (por exemplo, 'uvicorn', 'python -m ...', etc.).
# O `exec "$@"` garante que este script substitua seu próprio processo
# pelo processo da aplicação, o que é uma prática recomendada.
exec "$@"
