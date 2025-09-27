# Dockerfile Otimizado (Multi-stage)

# ==============================================================================
# Estágio 1: Builder
# - Instala dependências de build.
# - Baixa e compila todas as dependências do Python em "wheels".
# ==============================================================================
FROM python:3.11-slim-bookworm as builder

# Instala GCC/build-essential para compilar pacotes como numpy, scipy, etc.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /wheels

# Copia o arquivo de lock e cria os wheels.
# O uso de requirements.lock garante builds 100% reprodutíveis.
COPY requirements.lock .
RUN python -m pip install --upgrade pip && \
    python -m pip wheel -r requirements.lock --wheel-dir /wheels

# ==============================================================================
# Estágio 2: Runtime
# - Imagem final, leve e segura.
# - Instala os wheels pré-compilados, sem precisar do compilador.
# ==============================================================================
FROM python:3.11-slim-bookworm as runtime

# Instala apenas as dependências de runtime (curl, git).
RUN apt-get update && apt-get install -y --no-install-recommends curl git && rm -rf /var/lib/apt/lists/*

# Cria um usuário não-root para segurança.
RUN useradd -ms /bin/bash appuser

WORKDIR /app

# Copia os wheels do estágio de build e instala a partir deles.
# --no-index e --find-links garantem que só serão usados os pacotes locais.
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels -r /wheels/requirements.lock && rm -rf /wheels

# Copia o código da aplicação.
COPY . /app

# Ajusta permissões.
RUN chmod +x docker/entrypoint.sh && chown -R appuser:appuser /app

# Troca para o usuário não-root.
USER appuser

# Define o entrypoint e o comando padrão.
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]