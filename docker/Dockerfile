# Dockerfile - TrustShield Advanced (Versão 5.1.0-stable)
FROM python:3.10-slim

# Metadata
LABEL maintainer="TrustShield Team"
LABEL version="5.1.0-stable"
LABEL description="Sistema Avançado de Detecção de Fraudes"

# Variáveis de ambiente para otimização e logging
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Instalar dependências do sistema necessárias para as bibliotecas Python
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Criar um utilizador não-root para segurança
RUN useradd --create-home --shell /bin/bash trustshield
USER trustshield
WORKDIR /home/trustshield

# Copiar apenas o ficheiro de requisitos primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Instalar dependências Python como utilizador não-root
RUN pip install --user --no-cache-dir -r requirements.txt

# Copiar o resto do código da aplicação
COPY --chown=trustshield:trustshield . .

# Criar diretórios necessários que serão mapeados por volumes
RUN mkdir -p outputs/models logs mlruns

# Expor a porta do MLflow (embora o serviço principal seja outro, é uma boa prática)
EXPOSE 5000

# Comando padrão que será executado quando o contentor iniciar
CMD ["python", "src/models/train_fraud_model.py", "--config", "config/config.yaml", "--model", "all"]
