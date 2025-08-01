# Makefile - TrustShield Advanced (Versão 7.8.0 - Definitiva)
.PHONY: help install test lint format clean services-up services-down services-up-fresh train logs purge

# --- AJUDA ---
help:
	@echo "TrustShield Advanced - Comandos Disponíveis:"
	@echo ""
	@echo "--- GESTÃO DE SERVIÇOS ---"
	@echo "  services-up         - Inicia todos os serviços de suporte em segundo plano."
	@echo "  services-down       - Para todos os serviços de suporte."
	@echo "  services-up-fresh   - PARA, reconstrói e reinicia todos os serviços."
	@echo "  logs [service]      - Mostra os logs de um serviço (padrão: mlflow)."
	@echo ""
	@echo "--- PIPELINE & TAREFAS ---"
	@echo "  train [args]        - Executa o pipeline de treino (padrão: --model isolation_forest)."
	@echo ""
	@echo "--- LIMPEZA ---"
	@echo "  purge               - PARA TUDO e APAGA TODOS os dados (contêineres, volumes, redes)."
	@echo ""

# =====================================================================================
# === SEÇÃO DOCKER: O CORAÇÃO DA OPERAÇÃO ===
# =====================================================================================

# Passa o ficheiro .env explicitamente para todos os comandos
DOCKER_COMPOSE_CMD = docker compose --env-file ./.env -f docker/docker-compose.yml

services-up:
	@echo "🚀 Subindo todos os serviços de suporte e aguardando que fiquem saudáveis..."
	$(DOCKER_COMPOSE_CMD) up -d --wait
	@echo "✅ Todos os serviços estão prontos."

services-down:
	@echo "🛑 Parando todos os serviços..."
	$(DOCKER_COMPOSE_CMD) down --remove-orphans

services-up-fresh: services-down
	@echo "🧼 Reconstruindo a imagem e subindo todos os serviços do zero..."
	$(DOCKER_COMPOSE_CMD) up -d --build --force-recreate --remove-orphans

service ?= mlflow
logs:
	@echo "🔎 Acompanhando os logs do serviço: $(service)..."
	$(DOCKER_COMPOSE_CMD) logs -f $(service)

args ?= --model isolation_forest
train:
	@echo "🧠 Executando o pipeline de treino no ambiente unificado..."
	# CORREÇÃO: Usa 'exec' em vez de 'run'. 'exec' executa um comando num contentor JÁ A CORRER,
	# o que evita a condição de corrida e não precisa do entrypoint.sh.
	# Primeiro, garantimos que o serviço 'trustshield-api' está a correr.
	$(DOCKER_COMPOSE_CMD) up -d trustshield-api
	@echo "   Serviço da API está pronto. A executar a tarefa de treino dentro dele..."
	@echo "   Comando: python /home/trustshield/src/models/train_fraud_model.py $(args)"
	$(DOCKER_COMPOSE_CMD) exec trustshield-api python /home/trustshield/src/models/train_fraud_model.py $(args)

purge:
	@echo "🔥🔥🔥 ATENÇÃO: Parando e APAGANDO TODOS OS DADOS! 🔥🔥🔥"
	$(DOCKER_COMPOSE_CMD) down --volumes
	@echo "🧹 Limpando recursos do Docker..."
	docker builder prune -a -f
	docker system prune -f

# =====================================================================================
# === SEÇÃO DE DESENVOLVIMENTO LOCAL ===
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
	rm -rf .coverage htmlcov/ .pytest_cache/ cache/