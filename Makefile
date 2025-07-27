# Makefile - TrustShield Advanced (Versão 5.2.0-stable)
# Filosofia: Separa a gestão dos SERVIÇOS (que guardam dados) das TAREFAS (que rodam e terminam).
.PHONY: help install test lint format clean build train logs services-up services-down purge

# === AJUDA ===
help:
	@echo "TrustShield Advanced - Comandos Disponíveis:"
	@echo ""
	@echo "--- GESTÃO DE SERVIÇOS (PERSISTENTES) ---"
	@echo "  services-up    - Inicia os serviços de backend (Postgres, MinIO, MLflow UI) em segundo plano."
	@echo "  services-down  - Para os serviços de backend sem apagar os dados."
	@echo "  logs [service] - Mostra os logs de um serviço (ex: make logs service=mlflow). Padrão: trustshield-trainer."
	@echo ""
	@echo "--- PIPELINE & TAREFAS (EFÊMERAS) ---"
	@echo "  build          - Constrói (ou reconstrói) a imagem Docker da aplicação."
	@echo "  train          - Executa o pipeline de treino completo dentro do Docker. Requer 'services-up'."
	@echo ""
	@echo "--- LIMPEZA COMPLETA (DESTRUTIVO) ---"
	@echo "  purge          - PARA TUDO e APAGA TODOS os dados (contêineres, volumes, redes). Use com cuidado!"
	@echo ""
	@echo "--- DESENVOLVIMENTO LOCAL ---"
	@echo "  install        - Instalar dependências locais e pre-commit."
	@echo "  test           - Executar testes locais."
	@echo "  lint           - Verificar o estilo do código localmente."
	@echo "  format         - Formatar o código localmente."
	@echo "  clean          - Limpar ficheiros temporários do Python."

# =====================================================================================
# === SEÇÃO DOCKER: O CORAÇÃO DA OPERAÇÃO ===
# =====================================================================================

# --- GESTÃO DE SERVIÇOS (PERSISTENTES) ---

# Inicia os serviços de backend em segundo plano. Não inicia o treino.
# Estes serviços continuarão rodando até você usar 'services-down' ou 'purge'.
services-up:
	@echo "🚀 Subindo os serviços de backend (Postgres, MinIO, MLflow)..."
	docker compose -f docker/docker-compose.yml up -d postgres minio mlflow

# Para os serviços de backend. CRUCIAL: NÃO usa '--volumes', preservando os dados.
services-down:
	@echo "🛑 Parando os serviços de backend..."
	docker compose -f docker/docker-compose.yml down

# Mostra os logs. Pode especificar o serviço. Ex: make logs service=mlflow
service ?= trustshield-trainer
logs:
	@echo "🔎 Acompanhando os logs do serviço: $(service)..."
	docker compose -f docker/docker-compose.yml logs -f $(service)

# --- PIPELINE & TAREFAS (EFÊMERAS) ---

# Constrói a imagem Docker.
build:
	@echo "🛠️  Construindo a imagem 'trustshield-advanced:latest'..."
	docker build -t trustshield-advanced:latest -f docker/Dockerfile .

# Constrói a imagem do zero, ignorando qualquer cache. Útil para depuração.
build-fresh:
	@echo "🧼 Construindo a imagem 'trustshield-advanced:latest' do zero (sem cache)..."
	docker build --no-cache -t trustshield-advanced:latest -f docker/Dockerfile .

# Executa o treino como uma tarefa única. O contêiner é removido ao final (--rm).
# Isso permite que você rode o treino várias vezes sem acumular contêineres parados.
train:
	@echo "🧠 Executando o pipeline de treino..."
	docker compose -f docker/docker-compose.yml run --rm trustshield-trainer

# --- LIMPEZA COMPLETA (DESTRUTIVO) ---

# O antigo 'docker-stop'. Renomeado para 'purge' para deixar claro que é DESTRUTIVO.
# Use isto apenas quando quiser um reset completo do ambiente.
purge:
	@echo "🔥🔥🔥 ATENÇÃO: Parando todos os serviços e APAGANDO TODOS OS VOLUMES DE DADOS! 🔥🔥🔥"
	docker compose -f docker/docker-compose.yml down --volumes
	@echo "🧹 Limpando cache do Docker..."
	docker system prune -f


# =====================================================================================
# === SEÇÃO DE DESENVOLVIMENTO LOCAL (Não usa Docker) ===
# =====================================================================================
install:
	pip install -r requirements.txt
	pre-commit install

test:
	pytest tests/test_advanced.py -v

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/
