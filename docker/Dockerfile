# Dockerfile Unificado - TrustShield v9.2.0 (Robusto)
# Propósito: Construir uma imagem única e robusta com startup garantido.
FROM python:3.10-slim as builder

LABEL stage="builder"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Instala dependências de sistema, incluindo netcat para o entrypoint.
RUN apt-get update && apt-get install -y --no-install-recommends gcc netcat-openbsd && rm -rf /var/lib/apt/lists/*

# Cria o usuário da aplicação
RUN useradd --create-home --shell /bin/bash trustshield
WORKDIR /home/trustshield

# Copia e instala TODOS os requisitos
COPY --chown=trustshield:trustshield requirements.txt .
USER trustshield
RUN pip install --user -r requirements.txt

# --- Estágio Final ---
FROM python:3.10-slim as final

LABEL maintainer="TrustShield Team" version="9.2.0-robust"
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# Instala netcat, necessário para o script de entrypoint aguardar os serviços.
# É feito como root antes de mudar para o usuário da aplicação.
RUN apt-get update && apt-get install -y --no-install-recommends netcat-openbsd && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash trustshield
USER trustshield
WORKDIR /home/trustshield

COPY --from=builder --chown=trustshield:trustshield /home/trustshield/.local /home/trustshield/.local
ENV PATH="/home/trustshield/.local/bin:${PATH}"

# Copia o código-fonte, configuração e o script de entrypoint.
# O contexto do build é a raiz do projeto, então o caminho 'docker/entrypoint.sh' está correto.
COPY --chown=trustshield:trustshield src ./src
COPY --chown=trustshield:trustshield config ./config
COPY --chown=trustshield:trustshield docker/entrypoint.sh /usr/local/bin/entrypoint.sh

# Garante que o entrypoint seja executável.
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000

# Define o entrypoint que aguardará os serviços.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# O comando padrão (CMD) será passado como argumento para o entrypoint.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
