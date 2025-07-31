# Makefile - TrustShield Advanced (Versão 10.0.0 - com Pipeline de Forecast)
.PHONY: help install test lint format clean services-up services-down services-up-fresh train train-forecast logs purge build-force

# --- CONFIGURAÇÃO E PRÉ-REQUISITOS ---
DOCKER_COMPOSE_CMD = docker compose --env-file ./.env
API_SERVICE_NAME=trustshield-api

# Cores para o output
GREEN=\033[0;32m
YELLOW=\033[0;33m
NC=\033[0m # No Color

# Argumentos default para os pipelines, podem ser sobrescritos
# Ex: make train args="--outro-argumento"
train_args ?= --model isolation_forest
# Ex: make train-forecast ts_args="--model-type arima"
ts_args ?= --model-type prophet

# =====================================================================================
# === PIPELINES PRINCIPAIS ===
# =====================================================================================

train: build-force services-up process-data
	@echo "${YELLOW}🧠 Executando o pipeline de treino de deteção de anomalias...${NC}"
	$(DOCKER_COMPOSE_CMD) exec $(API_SERVICE_NAME) python /home/trustshield/src/models/train_fraud_model.py $(train_args)
	@echo "${GREEN}✅ Pipeline de treino de deteção de anomalias concluído.${NC}"

train-forecast: train aggregate-ts-data
	@echo "${YELLOW}📈 Executando o pipeline de treino do modelo de forecast...${NC}"
	$(DOCKER_COMPOSE_CMD) exec $(API_SERVICE_NAME) python /home/trustshield/src/models/train_ts_model.py $(ts_args)
	@echo "${GREEN}✅ Pipeline de forecast de séries temporais concluído com sucesso.${NC}"

# =====================================================================================
# === SUB-TARGETS E ORQUESTRAÇÃO ===
# =====================================================================================

process-data:
	@echo "${YELLOW}🔧 Executando o pipeline de processamento de dados...${NC}"
	$(DOCKER_COMPOSE_CMD) exec $(API_SERVICE_NAME) python /home/trustshield/src/data/make_dataset.py
	$(DOCKER_COMPOSE_CMD) exec $(API_SERVICE_NAME) python /home/trustshield/src/features/build_features.py
	@echo "${GREEN}✅ Processamento de dados concluído.${NC}"

aggregate-ts-data:
	@echo "${YELLOW}📊 Agregando dados para a série temporal...${NC}"
	$(DOCKER_COMPOSE_CMD) exec $(API_SERVICE_NAME) python /home/trustshield/src/data/aggregate_for_ts.py

# =====================================================================================
# === GESTÃO DE SERVIÇOS DOCKER ===
# =====================================================================================

build-force:
	@echo "${YELLOW}🏗️  Reconstruindo a imagem Docker '${API_SERVICE_NAME}' para garantir que as dependências estão atualizadas...${NC}"
	$(DOCKER_COMPOSE_CMD) build --no-cache $(API_SERVICE_NAME)

services-up:
	@echo "${YELLOW}🚀 Subindo todos os serviços de suporte e aguardando que fiquem saudáveis...${NC}"
	$(DOCKER_COMPOSE_CMD) up -d --wait
	@echo "${GREEN}✅ Todos os serviços estão prontos e saudáveis.${NC}"

services-down:
	@echo "${YELLOW}🛑 Parando todos os serviços...${NC}"
	$(DOCKER_COMPOSE_CMD) down --remove-orphans

services-up-fresh: build-force services-up

purge:
	@echo "${YELLOW}🔥🔥🔥 ATENÇÃO: Parando e APAGANDO TODOS OS DADOS! 🔥🔥🔥${NC}"
	$(DOCKER_COMPOSE_CMD) down --volumes
	@echo "${YELLOW}🧹 Limpando recursos do Docker (cache de build e imagens não utilizadas)...${NC}"
	docker builder prune -a -f
	docker system prune -f
	@echo "${GREEN}🧼 Limpeza completa.${NC}"

# =====================================================================================
# === COMANDOS DE DESENVOLVIMENTO ===
# =====================================================================================
service ?= mlflow
logs:
	@echo "🔎 Acompanhando os logs do serviço: $(service)... (Pressione Ctrl+C para sair)"
	$(DOCKER_COMPOSE_CMD) logs -f $(service)

install:
	pip install -r requirements.txt
	pre-commit install

test:
	pytest tests/ -v

lint:
	flake8 src/ tests/

format:
	black src/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
