# Makefile - TrustShield Advanced (Versão 8.1.0 - Otimizada e Robusta)
# Otimizado por Engenheiro de IA
.PHONY: help install test lint format clean services-up services-down services-up-fresh train logs purge

# --- CONFIGURAÇÕES GLOBAIS ---
# Centraliza o comando do Docker Compose para fácil manutenção.
COMPOSE_FILE := docker/docker-compose.yml
COMPOSE_CMD := docker compose -f $(COMPOSE_FILE)

# Define a meta 'help' como padrão quando 'make' é executado sem argumentos.
.DEFAULT_GOAL := help

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
	@echo "  train [args]        - Executa o pipeline de treino completo dentro do Docker (ex: make train args='--model lof')."
	@echo "  build [service]     - Reconstrói a imagem de um serviço específico (ex: make build service=trustshield-api)."
	@echo ""
	@echo "--- LIMPEZA COMPLETA (DESTRUTIVO) ---"
	@echo "  purge               - PARA TUDO e APAGA TODOS os dados (contêineres, volumes, redes). Use com cuidado!"


# --- GESTÃO DE SERVIÇOS (PERSISTENTES) ---
services-up:
	@echo "🚀 Iniciando todos os serviços do TrustShield em segundo plano..."
	$(COMPOSE_CMD) up -d

services-down:
	@echo "🛑 Parando todos os serviços do TrustShield..."
	$(COMPOSE_CMD) down

services-up-fresh:
	@echo "🔄 Reiniciando e reconstruindo os serviços do TrustShield..."
	$(COMPOSE_CMD) up -d --build --force-recreate --remove-orphans

service ?= trustshield-api
logs:
	@echo "🔎 Acompanhando os logs do serviço: $(service)..."
	$(COMPOSE_CMD) logs -f $(service)

build:
	@echo "🛠️  Reconstruindo a imagem do serviço: $(service)..."
	$(COMPOSE_CMD) build $(service)

# --- PIPELINE & TAREFAS (EFÊMERAS) ---
# Usa --entrypoint="" para executar comandos arbitrários no contêiner da API.
args ?= --model isolation_forest
train: services-up
	@echo "🧠 Executando o pipeline de treino no ambiente unificado..."
	@echo "   Comando: python /home/trustshield/src/models/train_fraud_model.py $(args)"
	$(COMPOSE_CMD) exec trustshield-api python /home/trustshield/src/models/train_fraud_model.py $(args)

# --- LIMPEZA COMPLETA (DESTRUTIVO) ---
# CORREÇÃO: Substituído '@read -r' por um snippet de shell compatível com /bin/sh.
purge:
	@echo "🔥🔥🔥 ATENÇÃO: Parando todos os serviços e APAGANDO TODOS OS VOLUMES DE DADOS! 🔥🔥🔥"
	@echo "🔥 Esta ação é irreversível. Pressione Enter para continuar ou Ctrl+C para cancelar."
	@read -p "Confirmação: " REPLY; \
	$(COMPOSE_CMD) down --volumes
	@echo "🧹 Limpando cache do builder do Docker..."
	docker builder prune -a -f
	@echo "🧹 Limpando outros recursos do Docker..."
	docker system prune -a -f
	@echo "✅ Limpeza completa finalizada."


# =====================================================================================
# === SEÇÃO DE DESENVOLVIMENTO LOCAL (Não usa Docker) ===
# =====================================================================================
install:
	pip install -r requirements.txt
	pre-commit install

test:
	pytest tests/test_advanced.py -v

lint:
	flake8 src