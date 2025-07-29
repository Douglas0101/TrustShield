# Makefile - TrustShield Advanced (Versão 5.2.0-stable)
# Filosofia: Separa a gestão dos SERVIÇOS (que guardam dados) das TAREFAS (que rodam e terminam).
.PHONY: help install test lint format clean build build-fresh train logs services-up services-down purge

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
	@echo "  build          - Constrói (ou reconstrói) a imagem Docker da aplicação usando cache."
	@echo "  build-fresh    - Constrói a imagem do zero, IGNORANDO O CACHE. Use para depuração."
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
services-up:
	@echo "🚀 Subindo os serviços de backend (Postgres, MinIO, MLflow)..."
	docker compose -f docker/docker-compose.yml up -d

services-down:
	@echo "🛑 Parando os serviços de backend..."
	docker compose -f docker/docker-compose.yml down

service ?= trustshield-trainer
logs:
	@echo "🔎 Acompanhando os logs do serviço: $(service)..."
	docker compose -f docker/docker-compose.yml logs -f $(service)

# --- PIPELINE & TAREFAS (EFÊMERAS) ---
build:
	@echo "🛠️  Construindo a imagem 'trustshield-advanced:latest'..."
	docker build -t trustshield-advanced:latest -f docker/Dockerfile .

# ESTA É A REGRA QUE FALTAVA. ELA FORÇA A RECONSTRUÇÃO.
build-fresh:
	@echo "🧼 Construindo a imagem 'trustshield-advanced:latest' do zero (sem cache)..."
	docker build --no-cache -t trustshield-advanced:latest -f docker/Dockerfile .

train:
	@echo "🧠 Executando o pipeline de treino..."
	docker compose -f docker/docker-compose.yml run --rm trustshield-trainer

# --- LIMPEZA COMPLETA (DESTRUTIVO) ---
purge:
	@echo "🔥🔥🔥 ATENÇÃO: Parando todos os serviços e APAGANDO TODOS OS VOLUMES DE DADOS! 🔥🔥🔥"
	docker compose -f docker/docker-compose.yml down --volumes
	@echo "🧹 Limpando cache do builder do Docker..."
	docker builder prune -a -f
	@echo "🧹 Limpando outros recursos do Docker..."
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