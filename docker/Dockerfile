# docker/Dockerfile
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dependências mínimas do sistema (mantenha leve)
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential gcc curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala dependências primeiro (melhor cache)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /app/requirements.txt

# Copia o código
COPY . /app

# Usuário não-root
RUN useradd -ms /bin/bash appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
