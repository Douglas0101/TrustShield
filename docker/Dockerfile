# ==============================================================================
# Dockerfile - TrustShield Enterprise Grade
# Versão: 10.0.0
#
# Otimizações e Melhores Práticas Implementadas:
# - Multi-stage builds para imagens de produção drasticamente menores.
# - Execução como usuário não-root para maior segurança.
# - Otimização do cache de layers para builds mais rápidos.
# - Variáveis de ambiente para boas práticas de CI/CD.
# - Remoção de código desnecessário (outputs não devem ser copiados).
# ==============================================================================

# --- Estágio 1: Builder ---
# Este estágio instala as dependências, incluindo as de compilação.
FROM python:3.10-slim-bookworm as builder

LABEL stage="builder"

# ENV para boas práticas
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Instala dependências de sistema necessárias para compilar algumas libs Python.
RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential && rm -rf /var/lib/apt/lists/*

# Cria um usuário não-root para a aplicação
RUN useradd --create-home --shell /bin/bash trustshield
WORKDIR /home/trustshield

# Copia apenas o requirements.txt primeiro para otimizar o cache.
# O build só re-executará este passo se o requirements.txt mudar.
COPY --chown=trustshield:trustshield requirements.txt .

# Instala as dependências como o usuário da aplicação
USER trustshield
RUN pip install --user -r requirements.txt


# --- Estágio 2: Final ---
# Este estágio cria a imagem final, que é muito mais leve.
FROM python:3.10-slim-bookworm as final

LABEL maintainer="TrustShield Team <contact@trustshield.dev>" version="10.0.0"

# Define as mesmas variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Instala o netcat, necessário para o script de health check no entrypoint.
# Faz isso antes de criar o usuário para ser executado como root.
RUN apt-get update && apt-get install -y --no-install-recommends netcat-openbsd && rm -rf /var/lib/apt/lists/*

# Cria o mesmo usuário não-root
RUN useradd --create-home --shell /bin/bash trustshield
USER trustshield
WORKDIR /home/trustshield

# Copia apenas as dependências instaladas do estágio 'builder'.
# Não trazemos o gcc ou outras ferramentas de build para a imagem final.
COPY --from=builder --chown=trustshield:trustshield /home/trustshield/.local /home/trustshield/.local

# Adiciona o diretório de binários do usuário ao PATH.
ENV PATH="/home/trustshield/.local/bin:${PATH}"

# Copia o código-fonte e a configuração da aplicação.
COPY --chown=trustshield:trustshield src ./src
COPY --chown=trustshield:trustshield config ./config
COPY --chown=trustshield:trustshield docker/entrypoint.sh ./docker/entrypoint.sh

# Garante que o entrypoint seja executável
RUN chmod +x ./docker/entrypoint.sh

EXPOSE 8000

# Define um comando padrão. Isto pode ser sobrescrito pelo docker-compose ou `docker run`.
# Aqui, o entrypoint é chamado para aguardar os serviços.
ENTRYPOINT ["./docker/entrypoint.sh"]
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
