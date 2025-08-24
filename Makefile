# =============================================================================
# TrustShield — Makefile “grande porte” (infra + MLOps + qualidade)
# =============================================================================
# Requisitos:
#   - Docker + Docker Compose
#   - Python 3.11 (se rodar etapas localmente) + requirements.txt
# Execução típica (da raiz do projeto):
#   make up           # sobe infra (Postgres, MinIO, MLflow, API, Dashboard)
#   make pipeline     # dados -> features -> train -> evaluate -> promote
#   make down         # derruba a stack
# =============================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ------------- Projeto / caminhos -------------
PROJECT_NAME := trustshield
COMPOSE_FILE := docker/docker-compose.yml
DC := docker compose -p $(PROJECT_NAME) -f $(COMPOSE_FILE)

PYTHON ?= python3
PIP    ?= pip

REQUIREMENTS := requirements.txt
CONFIG_YAML  := config/config.yaml

# Dados & artefatos
PRIMARY_PARQUET  := data/processed/primary_dataset.parquet
FEATURES_PARQUET := data/features/featured_dataset.parquet
MODELS_DIR       := outputs/models
DEFAULT_MODEL    := $(MODELS_DIR)/default_model.joblib

# ------------- Secrets (para rodar local com MLflow/MinIO) -------------
POSTGRES_PASSWORD_FILE := secrets/postgres_password.txt
MINIO_USER_FILE        := secrets/minio_root_user.txt
MINIO_PASS_FILE        := secrets/minio_root_password.txt

# Endpoints locais (containers expõem as portas no host)
export MLFLOW_TRACKING_URI      := http://127.0.0.1:5000
export MLFLOW_S3_ENDPOINT_URL   := http://127.0.0.1:9000

# ------------- Aparência -------------
GREEN  := \033[1;32m
CYAN   := \033[1;36m
YELLOW := \033[1;33m
RED    := \033[1;31m
NC     := \033[0m

# ------------- Utilitários -------------
define _banner
	@echo -e "$(CYAN)[TrustShield]$(NC) $1"
endef

define _export_minio_env
	@if [[ -f "$(MINIO_USER_FILE)" && -f "$(MINIO_PASS_FILE)" ]]; then \
	  export AWS_ACCESS_KEY_ID="$$(cat $(MINIO_USER_FILE))"; \
	  export AWS_SECRET_ACCESS_KEY="$$(cat $(MINIO_PASS_FILE))"; \
	else \
	  echo -e "$(RED)[ERRO] Secrets do MinIO ausentes em secrets/*.txt$(NC)"; exit 1; \
	fi
endef

define _wait_http
	@url="$$1"; name="$$2"; \
	$(call _banner,"Aguardando $$name em $$url ..."); \
	for i in {1..60}; do \
	  if curl -fsS "$$url" >/dev/null 2>&1; then \
	    echo -e "$(GREEN)[OK]$$name pronto$(NC)"; exit 0; \
	  fi; \
	  sleep 2; \
	done; \
	echo -e "$(RED)[FALHA] $$name não respondeu a tempo$(NC)"; exit 1
endef

# =============================================================================
# Ajuda
# =============================================================================
.PHONY: help
help: ## Mostra esta ajuda
	@echo -e "$(GREEN)Alvos principais$(NC):"
	@grep -E '^[a-zA-Z0-9\._-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*##/: /' | \
	awk 'BEGIN {FS = ": "}; {printf "  $(CYAN)%-22s$(NC) %s\n", $$1, $$2}'

# =============================================================================
# Infraestrutura (Docker Compose)
# =============================================================================
.PHONY: up down ps restart logs build rebuild
up: ## Sobe a stack (build + up -d) e aguarda MLflow/API
	$(call _banner,"Subindo infraestrutura Docker...")
	DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 $(DC) up -d --build
	$(call _wait_http,"http://127.0.0.1:5000","MLflow")
	$(call _wait_http,"http://127.0.0.1:8000/healthz","API")

down: ## Derruba a stack completa (containers, volumes e órfãos)
	$(DC) down --volumes --remove-orphans

ps: ## Lista serviços da stack
	$(DC) ps

restart: ## Reinicia a stack
	$(DC) restart

logs: ## Mostra logs em tempo real de um serviço. Ex.: make logs service=api
	@if [ -z "$(service)" ]; then echo "Uso: make logs service=<nome>"; exit 1; fi
	$(DC) logs -f $(service)

build: ## (Re)constrói imagens
	DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 $(DC) build

rebuild: ## Limpa cache e refaz build
	DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 $(DC) build --no-cache

# =============================================================================
# Setup local (fora dos contêineres)
# =============================================================================
.PHONY: install init-env check-secrets
install: ## Instala dependências locais (usa requirements.txt)
	$(call _banner,"Instalando dependências locais...")
	$(PIP) install -U pip
	$(PIP) install -r $(REQUIREMENTS)

init-env: ## Gera .env a partir de secrets (opcional para desenvolvedor)
	@echo "POSTGRES_PASSWORD=$$(cat $(POSTGRES_PASSWORD_FILE))" > .env
	@echo "MINIO_ROOT_USER=$$(cat $(MINIO_USER_FILE))"       >> .env
	@echo "MINIO_ROOT_PASSWORD=$$(cat $(MINIO_PASS_FILE))"  >> .env
	@echo -e "$(GREEN)[OK] .env gerado a partir de secrets$(NC)"

check-secrets: ## Verifica se secrets estão presentes
	@test -f $(POSTGRES_PASSWORD_FILE)
	@test -f $(MINIO_USER_FILE)
	@test -f $(MINIO_PASS_FILE)
	@echo -e "$(GREEN)[OK] Secrets existentes em ./secrets$(NC)"

# =============================================================================
# Pipeline MLOps (scripts do repositório)
# =============================================================================
# Módulos utilizados:
#  - build_features.py (engenharia de features) ...................... src/features/  ✔  :contentReference[oaicite:11]{index=11}
#  - optimization.py (HPO com Optuna/Ray) ............................ src/models/    ✔  :contentReference[oaicite:12]{index=12}
#  - evaluate_models.py (comparação de modelos) ...................... src/models/    ✔  :contentReference[oaicite:13]{index=13}
#  - train_fraud_model.py (treino/empacotamento) ..................... src/models/    ✔  :contentReference[oaicite:14]{index=14}
#  - validation.py (quality gates / drift) ........................... src/models/    ✔  :contentReference[oaicite:15]{index=15}

.PHONY: data features train optimize evaluate promote pipeline retrain validate
data: ## Gera dataset primário (src/data/make_dataset.py) -> $(PRIMARY_PARQUET)
	$(call _banner,"Gerando dataset primário...")
	@set -euo pipefail; \
	if [ -f src/data/make_dataset.py ]; then \
	  $(PYTHON) src/data/make_dataset.py || $(PYTHON) -m src.data.make_dataset; \
	else \
	  echo -e "$(YELLOW)[AVISO] src/data/make_dataset.py não encontrado; pulando etapa$(NC)"; \
	fi

features: ## Engenharia de features -> $(FEATURES_PARQUET)
	$(call _banner,"Criando features...")
	@set -euo pipefail; \
	$(PYTHON) src/features/build_features.py

train: check-secrets ## Treina um modelo (usa MLflow/MinIO locais)
	$(call _banner,"Treinando modelo...")
	@set -euo pipefail; \
	$(_export_minio_env); \
	$(PYTHON) src/models/train_fraud_model.py

optimize: check-secrets ## Otimização de hiperparâmetros (Optuna/Ray)
	$(call _banner,"Otimização de hiperparâmetros (Optuna)...")
	@set -euo pipefail; \
	$(_export_minio_env); \
	$(PYTHON) src/models/optimization.py --data $(FEATURES_PARQUET) --config $(CONFIG_YAML)

evaluate: ## Avalia os modelos mais recentes (top-5)
	$(call _banner,"Avaliando modelos mais recentes...")
	@set -euo pipefail; \
	models=$$(ls -1t $(MODELS_DIR)/*.joblib 2>/dev/null | head -n 5); \
	if [ -z "$$models" ]; then echo -e "$(RED)Nenhum modelo encontrado em $(MODELS_DIR)$(NC)"; exit 1; fi; \
	$(PYTHON) src/models/evaluate_models.py --data $(FEATURES_PARQUET) --models $$models --config $(CONFIG_YAML)

promote: ## Promove o melhor/mais recente para $(DEFAULT_MODEL)
	$(call _banner,"Promovendo modelo de produção...")
	@set -euo pipefail; \
	cand=$$(ls -1t $(MODELS_DIR)/isolation_forest_optimized_*.joblib 2>/dev/null | head -n 1); \
	if [ -z "$$cand" ]; then cand=$$(ls -1t $(MODELS_DIR)/isolation_forest_*.joblib 2>/dev/null | head -n 1); fi; \
	if [ -z "$$cand" ]; then echo -e "$(RED)Nenhum artefato para promover$(NC)"; exit 1; fi; \
	cp -f "$$cand" $(DEFAULT_MODEL); \
	echo -e "$(GREEN)[OK] Promovido: $$cand -> $(DEFAULT_MODEL)$(NC)"

validate: ## Executa quality gates (ex.: drift) com Evidently (opcional)
	$(call _banner,"Validação / Quality Gates...")
	@set -euo pipefail; \
	$(PYTHON) src/models/validation.py \
	  --data $(FEATURES_PARQUET) \
	  --reference data/features/featured_dataset.parquet \
	  --types drift_detection

pipeline: ## Pipeline end-to-end: data -> features -> train -> evaluate -> promote
	$(MAKE) data
	$(MAKE) features
	$(MAKE) train
	$(MAKE) evaluate
	$(MAKE) promote

retrain: ## Pipeline com HPO: data -> features -> optimize -> train -> evaluate -> promote
	$(MAKE) data
	$(MAKE) features
	$(MAKE) optimize
	$(MAKE) train
	$(MAKE) evaluate
	$(MAKE) promote

# =============================================================================
# Serviços locais (fora/ao lado dos contêineres) — útil para dev
# =============================================================================
.PHONY: serve-api serve-dashboard
serve-api: ## Sobe API local (uvicorn) usando o modelo promovido
	$(call _banner,"Subindo API local (uvicorn)...")
	@set -euo pipefail; \
	if [ ! -f "$(DEFAULT_MODEL)" ]; then echo -e "$(YELLOW)$(DEFAULT_MODEL) não existe. Rode 'make promote'.$(NC)"; exit 1; fi; \
	MODEL_PATH="$(DEFAULT_MODEL)" uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1

serve-dashboard: ## Sobe o dashboard local (Streamlit)
	$(call _banner,"Abrindo Dashboard (Streamlit)...")
	@set -euo pipefail; \
	streamlit run src/dashboard/app.py

# =============================================================================
# Qualidade de código e testes
# =============================================================================
.PHONY: lint fmt typecheck test cov
lint: ## Lint com flake8
	flake8 src tests

fmt: ## Formata com black
	black src tests

typecheck: ## Type-check com mypy (parcial)
	mypy src || true

test: ## Testes (pytest)
	pytest -q

cov: ## Testes com cobertura
	pytest --cov=src --cov-report=term-missing

# =============================================================================
# Limpeza
# =============================================================================
.PHONY: clean deepclean nuke
clean: ## Remove caches/artefatos leves
	$(call _banner,"Limpando caches...")
	find . -type d -name "__pycache__" -exec rm -rf {} + || true
	rm -rf .pytest_cache .mypy_cache || true

deepclean: clean ## Limpa saídas de pipeline (NÃO remove modelos)
	$(call _banner,"Limpando saídas do pipeline...")
	rm -rf outputs/interpretations outputs/validation outputs/temp_explanations || true

nuke: ## ⚠️ Remove TUDO do projeto no Docker (containers, volumes, imagens)
	@read -p "Confirma limpeza radical? (y/N) " ans; \
	if [[ "$$ans" == "y" || "$$ans" == "Y" ]]; then \
	  $(DC) down -v --rmi local --remove-orphans; \
	  docker builder prune -a -f; \
	  docker system prune -f; \
	  docker network prune -f; \
	  docker network rm trustshield trustshield_default docker_trustshield-net 2>/dev/null || true; \
	  echo -e "$(YELLOW)[CUIDADO] Volumes/imagens removidos$(NC)"; \
	else \
	  echo "Cancelado."; \
	fi