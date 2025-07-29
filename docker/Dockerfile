# Dockerfile Unificado - TrustShield v9.1.0 (Corrigido)
# Propósito: Construir uma imagem única e robusta a partir do código-fonte e dependências.
FROM python:3.10-slim as builder

LABEL stage="builder"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Instala dependências de sistema
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Cria o usuário da aplicação
RUN useradd --create-home --shell /bin/bash trustshield
WORKDIR /home/trustshield

# Copia e instala TODOS os requisitos
COPY --chown=trustshield:trustshield requirements.txt .
USER trustshield
RUN pip install --user -r requirements.txt

# --- Estágio Final ---
FROM python:3.10-slim as final

LABEL maintainer="TrustShield Team" version="9.1.0-unified"
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

RUN useradd --create-home --shell /bin/bash trustshield
USER trustshield
WORKDIR /home/trustshield

COPY --from=builder --chown=trustshield:trustshield /home/trustshield/.local /home/trustshield/.local
ENV PATH="/home/trustshield/.local/bin:${PATH}"

# Copia o código-fonte e a configuração.
COPY --chown=trustshield:trustshield src ./src
COPY --chown=trustshield:trustshield config ./config

# CORREÇÃO: A linha 'COPY outputs' foi REMOVIDA.
# A pasta 'outputs' é gerenciada por um volume no docker-compose.yml,
# que é a abordagem correta para dados gerados em tempo de execução.

EXPOSE 8000

# O comando padrão é iniciar a API. O Makefile irá sobrescrever isso para o treino.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]