# Makefile - TrustShield Advanced (Versão 7.0.0 - Compose V2 Fix)
# CORREÇÃO: Usa 'docker compose' (V2, com espaço) em vez do obsoleto 'docker-compose' (V1, com hífen).
.PHONY: help install test lint format clean services-up services-down services-up-fresh train logs purge

# === AJUDA ===
help:
	@echo "TrustShield Advanced - Comandos Disponíveis:"
	@echo ""
	@echo "--- GESTÃO DE SERVIÇOS (PERSISTENTES) ---"
	@echo "  services-up         - Inicia todos os serviços (API, Postgres, MinIO, MLflow) em segundo plano."
	@echo "  services-down       - Para os serviços de backend sem apagar os dados."
	@echo "  services-up-fresh   - Reconstrói a imagem unificada e reinicia os serviços. Use para aplicar grandes mudanças."
	@echo "  logs [service]      - Mostra os logs de um serviço (ex: make logs service=mlflow). Padrão: trustshield-api."
	@echo ""
	@echo "--- PIPELINE & TAREFAS (EFÊMERAS) ---"
	@echo "  train [args]        - Executa o pipeline de treino completo dentro do Docker (ex: make train args='--model lof'). Requer 'services-up'."
	@echo ""
	@echo "--- LIMPEZA COMPLETA (DESTRUTIVO) ---"
	@echo "  purge               - PARA TUDO e APAGA TODOS os dados (contêineres, volumes, redes). Use com cuidado!"
	@echo ""
	@echo "--- DESENVOLVIMENTO LOCAL ---"
	@echo "  install             - Instalar dependências locais e pre-commit."
	@echo "  test                - Executar testes locais."
	@echo "  lint                - Verificar o estilo do código localmente."
	@echo "  format              - Formatar o código localmente."
	@echo "  clean               - Limpar ficheiros temporários do Python."

# =====================================================================================
# === SEÇÃO DOCKER: O CORAÇÃO DA OPERAÇÃO ===
# =====================================================================================

# ATUALIZAÇÃO: Trocado 'docker-compose' por 'docker compose' em todos os comandos.
services-up:
	@echo "🚀 Subindo todos os serviços (API, Postgres, MinIO, MLflow)..."
	docker compose -f docker-compose.yml up -d --remove-orphans

services-down:
	@echo "🛑 Parando todos os serviços..."
	docker compose -f docker-compose.yml down --remove-orphans

services-up-fresh:
	@echo "🧼 Reconstruindo a imagem unificada e subindo todos os serviços do zero..."
	docker compose -f docker-compose.yml up -d --build --force-recreate --remove-orphans

service ?= trustshield-api
logs:
	@echo "🔎 Acompanhando os logs do serviço: $(service)..."
	docker compose -f docker-compose.yml logs -f $(service)

# --- PIPELINE & TAREFAS (EFÊMERAS) ---
args ?= --model isolation_forest
make-dataset:
	@echo "Creating the dataset..."
	@echo "Command: python /home/trustshield/src/data/make_dataset.py"
	docker compose -f docker-compose.yml run --rm trustshield-api python /home/trustshield/src/data/make_dataset.py

build-features:
	@echo "🛠️  Executando a engenharia de features no ambiente unificado..."
	@echo "   Comando: python /home/trustshield/src/features/build_features.py"
	docker compose -f docker-compose.yml run --rm trustshield-api python /home/trustshield/src/features/build_features.py

train:
	@echo "🧠 Executando o pipeline de treino no ambiente unificado..."
	@echo "   Comando: python /home/trustshield/src/models/train_fraud_model.py $(args)"
	docker compose -f docker-compose.yml run --rm trustshield-api python /home/trustshield/src/models/train_fraud_model.py $(args)

# --- LIMPEZA COMPLETA (DESTRUTIVO) ---
purge:
	@echo "🔥🔥🔥 ATENÇÃO: Parando todos os serviços e APAGANDO TODOS OS VOLUMES DE DADOS! 🔥🔥🔥"
	docker compose -f docker-compose.yml down --volumes
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