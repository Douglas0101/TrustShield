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

# Agora, execute o comando principal que foi passado para o container
# (por exemplo, 'uvicorn', 'python -m ...', etc.).
# O `exec "$@"` garante que este script substitua seu próprio processo
# pelo processo da aplicação, o que é uma prática recomendada.
exec "$@"