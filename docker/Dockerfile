# Dockerfile - TrustShield Advanced
FROM python:3.9-slim

# Metadata
LABEL maintainer="TrustShield Team"
LABEL version="4.0.0-enterprise"
LABEL description="Sistema Avançado de Detecção de Fraudes"

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Criar usuário não-root
RUN useradd --create-home --shell /bin/bash trustshield
USER trustshield
WORKDIR /home/trustshield

# Copiar requirements
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --user --no-cache-dir -r requirements.txt

# Copiar código
# Linha correta a ser adicionada
COPY --chown=trustshield:trustshield . .

# Criar diretórios necessários
RUN mkdir -p outputs/models logs mlruns

# Expor porta MLflow
EXPOSE 5000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Comando padrão
CMD ["python", "-m", "src.models.train_fraud_model", "--config", "config/config.yaml", "--model", "all"]
