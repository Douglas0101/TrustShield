# Dockerfile Final (CPU) - TrustShield Advanced v7.0.4-cpu
# Este Dockerfile é otimizado para ambientes baseados em CPU, como o Intel i3.
# Ele é leve, seguro e contém todas as dependências necessárias para a execução do seu projeto.

# 1. Imagem Base: Leve e oficial do Python.
FROM python:3.10-slim

# 2. Metadata: Boas práticas para identificar a imagem.
LABEL maintainer="TrustShield Team"
LABEL version="7.0.4-cpu"
LABEL description="Sistema Avançado de Detecção de Fraudes (CPU Build)"

# 3. Variáveis de Ambiente: Otimização e logging.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 4. Instalação de Dependências do Sistema (INCLUINDO O GIT)
# Instala o 'git' para que o MLflow possa rastrear os commits.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# 5. Segurança: Criar e usar um utilizador não-root.
RUN useradd --create-home --shell /bin/bash trustshield
USER trustshield
WORKDIR /home/trustshield

# 6. Otimização de Cache: Copiar apenas 'requirements.txt' primeiro.
COPY --chown=trustshield:trustshield requirements.txt .

# 7. Instalação de Dependências Python
# Instala as dependências como o usuário 'trustshield'
RUN pip install --user --no-cache-dir -r requirements.txt

# 8. Adiciona o diretório de binários do usuário ao PATH
ENV PATH="/home/trustshield/.local/bin:${PATH}"

# 9. Copiar o entrypoint e torná-lo executável
COPY --chown=trustshield:trustshield docker/entrypoint.sh /home/trustshield/entrypoint.sh
RUN chmod +x /home/trustshield/entrypoint.sh

# 10. Cria diretórios para volumes para evitar problemas de permissão
RUN mkdir -p outputs logs

# 11. Copiar o restante do código-fonte
COPY --chown=trustshield:trustshield src/ ./src
COPY --chown=trustshield:trustshield config/ ./config

# 12. Ponto de Entrada: Executa o script de entrada.
ENTRYPOINT ["/home/trustshield/entrypoint.sh"]