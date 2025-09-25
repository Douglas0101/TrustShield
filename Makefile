# ============================================================================
# TrustShield — Makefile v3.0
# ---------------------------------------------------------------------------
# Orquestração completa do ambiente de MLOps (Docker) e utilidades locais
# após as otimizações recentes no pipeline de treinamento/monitoramento.
# ============================================================================

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# 🚢 Configuração Docker / Compose
# ---------------------------------------------------------------------------
PROJECT        ?= trustshield
COMPOSE_FILE   ?= docker/docker-compose.yml
DC             := DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker compose -p $(PROJECT) -f $(COMPOSE_FILE)

API_SERVICE    ?= trustshield-api
DASH_SERVICE   ?= trustshield-dashboard
MLFLOW_SERVICE ?= mlflow
MINIO_SERVICE  ?= minio
POSTGRES_SERVICE ?= postgres

API_HOST_PORT     ?= 8000
DASH_HOST_PORT    ?= 8501
MLFLOW_HOST_PORT  ?= 5000
MINIO_HOST_PORT   ?= 9000
MINIO_CONSOLE_PORT?= 9001

HOST_API_URL      := http://localhost:$(API_HOST_PORT)
HOST_DASH_URL     := http://localhost:$(DASH_HOST_PORT)
HOST_MLFLOW_URL   := http://localhost:$(MLFLOW_HOST_PORT)
HOST_MINIO_URL    := http://localhost:$(MINIO_HOST_PORT)
HOST_MINIO_CONSOLE_URL := http://localhost:$(MINIO_CONSOLE_PORT)

RUN_API := $(DC) exec $(API_SERVICE) /app/docker/entrypoint.sh

NO_CACHE ?= 0
PULL     ?= 0
BUILD_ARGS :=
ifeq ($(NO_CACHE),1)
  BUILD_ARGS += --no-cache
endif
ifeq ($(PULL),1)
  BUILD_ARGS += --pull always
endif

# ---------------------------------------------------------------------------
# 📦 Caminhos de artefatos e configurações do pipeline
# ---------------------------------------------------------------------------
FEATURED_DATASET ?= data/features/featured_dataset.parquet
PRIMARY_DATASET  ?= data/processed/primary_dataset.parquet
MODEL_ARTIFACT   ?= outputs/models/default_model.joblib
OPT_MODELS_DIR   ?= outputs/models
CONFIG_FILE      ?= config/config.yaml

PYTHON ?= python3
PIP    ?= pip3
VENV   ?= .venv
VENV_BIN := $(VENV)/bin

# ---------------------------------------------------------------------------
# 🆘 Utilitário de ajuda (grep de comentários com ##)
# ---------------------------------------------------------------------------
define PRINT_HELP
Available targets:
  docker ---------------------------------------------------------------
endef
export PRINT_HELP

.PHONY: help
help: ## Mostra este menu de ajuda
	@echo "$(PRINT_HELP)"
	@grep -hE '^[a-zA-Z0-9_.-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'
	@printf "\nVariáveis principais:\n  PROJECT=%s\n  COMPOSE_FILE=%s\n  API_PORT=%s\n  MLFLOW_PORT=%s\n" "$(PROJECT)" "$(COMPOSE_FILE)" "$(API_HOST_PORT)" "$(MLFLOW_HOST_PORT)"

# ---------------------------------------------------------------------------
# ⚙️ Preparação do ambiente Docker
# ---------------------------------------------------------------------------
.PHONY: check
check: ## Valida pré-requisitos locais (docker / compose / arquivos)
	command -v docker >/dev/null 2>&1 || { echo "ERRO: docker não encontrado"; exit 1; }
	docker compose version >/dev/null 2>&1 || { echo "ERRO: docker compose v2 não encontrado"; exit 1; }
	[ -f $(COMPOSE_FILE) ] || { echo "ERRO: $(COMPOSE_FILE) não existe"; exit 1; }
	$(DC) config >/dev/null || { echo "ERRO: 'docker compose config' falhou"; exit 1; }
	@echo "OK: pré-checagens passaram."

.PHONY: build
build: check ## Constrói as imagens dos serviços
	$(DC) build $(BUILD_ARGS)

.PHONY: up
up: build ## Sobe todos os serviços e aguarda API healthy
	$(DC) up -d
	$(MAKE) wait

.PHONY: down
down: ## Derruba os serviços (mantém volumes)
	$(DC) down --remove-orphans

.PHONY: restart
restart: ## Reinicia todos os serviços
	$(MAKE) down
	$(MAKE) up

.PHONY: rebuild
rebuild: ## Recria imagens sem cache e reinicia
	$(MAKE) build NO_CACHE=1
	$(DC) up -d
	$(MAKE) wait

.PHONY: ps
ps: ## Lista serviços ativos e status
	$(DC) ps

.PHONY: top
top: ## Exibe processos/recursos dos serviços
	$(DC) top

.PHONY: logs
logs: ## Segue logs de todos os serviços
	$(DC) logs -f

.PHONY: api-logs
api-logs: ## Logs apenas da API
	$(DC) logs -f $(API_SERVICE)

.PHONY: dash-logs
dash-logs: ## Logs do dashboard
	$(DC) logs -f $(DASH_SERVICE)

.PHONY: mlflow-logs
mlflow-logs: ## Logs do MLflow
	$(DC) logs -f $(MLFLOW_SERVICE)

.PHONY: minio-logs
minio-logs: ## Logs do MinIO
	$(DC) logs -f $(MINIO_SERVICE)

.PHONY: wait
wait: ## Aguarda a API ficar healthy (healthcheck + endpoint)
	cid=$$($(DC) ps -q $(API_SERVICE)); \
	[ -n "$$cid" ] || { echo "ERRO: container da API não encontrado."; exit 1; }; \
	for i in $$(seq 1 60); do \
	  s=$$(docker inspect -f '{{.State.Health.Status}}' $$cid 2>/dev/null || echo "unknown"); \
	  echo "  tentativa $$i: $$s"; \
	  [ "$$s" = "healthy" ] && break; \
	  sleep 2; \
	done; \
	[ "$$s" = "healthy" ] || { echo "ERRO: API não ficou healthy a tempo."; exit 1; }; \
	echo "Container está healthy. Esperando 2s..."; \
	sleep 2; \
	curl -fsS $(HOST_API_URL)/healthz >/dev/null && echo "OK: /healthz"

.PHONY: health
health: ## Mostra status detalhado da API + endpoint /healthz
	cid=$$($(DC) ps -q $(API_SERVICE)); \
	[ -n "$$cid" ] || { echo "ERRO: container da API não encontrado."; exit 1; }; \
	docker inspect -f 'Status={{.State.Status}} Health={{if .State.Health}}{{.State.Health.Status}}{{else}}(sem healthcheck){{end}}' $$cid
	@echo
	@curl -i --max-time 3 $(HOST_API_URL)/healthz || true

.PHONY: status
status: ## Exibe resumo de portas e URLs úteis
	@echo "API.............: $(HOST_API_URL)"
	@echo "Dashboard.......: $(HOST_DASH_URL)"
	@echo "MLflow..........: $(HOST_MLFLOW_URL)"
	@echo "MinIO...........: $(HOST_MINIO_URL)"
	@echo "MinIO Console...: $(HOST_MINIO_CONSOLE_URL)"

.PHONY: env
env: ## Imprime variáveis do ambiente de execução
	@echo "PROJECT=$(PROJECT)"
	@echo "COMPOSE_FILE=$(COMPOSE_FILE)"
	@echo "API=$(API_SERVICE) DASH=$(DASH_SERVICE) MLFLOW=$(MLFLOW_SERVICE) MINIO=$(MINIO_SERVICE)"
	@echo "HOSTS: API=$(HOST_API_URL) DASH=$(HOST_DASH_URL) MLFLOW=$(HOST_MLFLOW_URL)"
	@echo "BUILD_ARGS='$(BUILD_ARGS)'"

.PHONY: api-shell
api-shell: ## Abre shell no container da API
	cid=$$($(DC) ps -q $(API_SERVICE)); [ -n "$$cid" ] && (docker exec -it $$cid bash || docker exec -it $$cid sh)

.PHONY: dash-shell
dash-shell: ## Abre shell no container do dashboard
	cid=$$($(DC) ps -q $(DASH_SERVICE)); [ -n "$$cid" ] && (docker exec -it $$cid bash || docker exec -it $$cid sh)

.PHONY: mlflow-shell
mlflow-shell: ## Abre shell no container do MLflow
	cid=$$($(DC) ps -q $(MLFLOW_SERVICE)); [ -n "$$cid" ] && (docker exec -it $$cid bash || docker exec -it $$cid sh)

.PHONY: minio-shell
minio-shell: ## Abre shell no container do MinIO
	cid=$$($(DC) ps -q $(MINIO_SERVICE)); [ -n "$$cid" ] && docker exec -it $$cid sh

# ---------------------------------------------------------------------------
# 🧹 Limpeza de ambiente e artefatos
# ---------------------------------------------------------------------------
.PHONY: clean
clean: down ## Remove containers e volumes nomeados do projeto
	vols=$$(docker volume ls -q | grep -E '^$(PROJECT)_' || true); \
	[ -n "$$vols" ] && docker volume rm $$vols || echo "Sem volumes nomeados para remover."
	@$(MAKE) cache-clean

.PHONY: prune
prune: ## Limpeza geral do host Docker (imagens/volumes/parados)
	@echo "ATENÇÃO: limpa imagens/containers/volumes não usados (host inteiro)"
	docker system prune -af --volumes

.PHONY: cache-clean
cache-clean: ## Remove cache local de dados
	rm -f cache/data_cache.pkl

.PHONY: models-clean
models-clean: ## Remove modelos gerados (mantém default_model)
	find $(OPT_MODELS_DIR) -maxdepth 1 -type f -name 'isolation_forest_*.joblib' -delete

.PHONY: mlruns-clean
mlruns-clean: ## Remove experimentos MLflow locais
	rm -rf mlruns

# ---------------------------------------------------------------------------
# 🧪 Testes e utilidades locais
# ---------------------------------------------------------------------------
.PHONY: venv
venv: ## Cria ambiente virtual local (.venv)
	[ -d $(VENV) ] || $(PYTHON) -m venv $(VENV)

.PHONY: deps
deps: venv ## Instala dependências no ambiente virtual
	$(VENV_BIN)/pip install --upgrade pip
	$(VENV_BIN)/pip install -r requirements.txt

.PHONY: test
test: ## Executa pytest localmente
	$(PYTHON) -m pytest

.PHONY: test-docker
test-docker: ## Executa pytest dentro do container da API
	$(RUN_API) python -m pytest

# ---------------------------------------------------------------------------
# 🔁 Pipeline de dados e modelos (executado dentro do serviço da API)
# ---------------------------------------------------------------------------
.PHONY: data
data: ## Executa ingestão/limpeza de dados (src.data.make_dataset)
	$(RUN_API) python -m src.data.make_dataset

.PHONY: features
features: ## Executa engenharia de features (src.features.build_features)
	$(RUN_API) python -m src.features.build_features

.PHONY: train
train: ## Treina a suíte completa de modelos otimizados
	$(RUN_API) python -m src.models.train_fraud_model

.PHONY: eval
eval: ## Avalia modelos utilizando dados de features
	$(RUN_API) python -m src.models.evaluate_models --data $(FEATURED_DATASET) --models $(MODEL_ARTIFACT)

.PHONY: optimize
optimize: ## Executa otimização de hiperparâmetros (Optuna)
	$(RUN_API) python -m src.models.optimization --data $(FEATURED_DATASET) --config $(CONFIG_FILE)

.PHONY: validate
validate: ## Roda validações de qualidade (dados/modelo)
	$(RUN_API) python -m src.models.validation --data $(FEATURED_DATASET) --model $(MODEL_ARTIFACT) --reference $(PRIMARY_DATASET)

.PHONY: interpret
interpret: ## Gera interpretações do modelo (SHAP)
	$(RUN_API) python -m src.models.interpretation --model $(MODEL_ARTIFACT) --data $(FEATURED_DATASET)

.PHONY: promote
promote: ## Promove o modelo otimizado mais recente para default_model.joblib
	$(RUN_API) python -c "from pathlib import Path; import shutil, sys; p=Path('$(OPT_MODELS_DIR)'); models=sorted(p.glob('isolation_forest_optimized_*.joblib'), key=lambda x: x.stat().st_mtime, reverse=True); (models and (shutil.copy2(models[0], p / 'default_model.joblib'), print(f'Promovido: {models[0].name}'))) or sys.exit('Nenhum modelo otimizado encontrado.')"

.PHONY: reload
reload: ## Reinicia o serviço da API para carregar novo modelo
	$(DC) restart $(API_SERVICE)

.PHONY: smoke
smoke: ## Health-check /status da API hospedada
	curl -fsS $(HOST_API_URL)/healthz >/dev/null && echo "OK: /healthz"
	curl -fsS $(HOST_API_URL)/status >/dev/null && echo "OK: /status"

.PHONY: cycle
cycle: ## Executa o pipeline completo de ponta a ponta
	$(MAKE) data
	$(MAKE) features
	$(MAKE) train
	$(MAKE) promote
	$(MAKE) eval
	$(MAKE) optimize
	$(MAKE) validate
	$(MAKE) interpret
	$(MAKE) reload
	$(MAKE) wait
	$(MAKE) smoke

.PHONY: quick
quick: ## Limpa ambiente e executa ciclo completo do zero
	$(MAKE) clean
	$(MAKE) up
	$(MAKE) cycle

# ---------------------------------------------------------------------------
# 🛠️ Comandos locais adicionais para desenvolvedores
# ---------------------------------------------------------------------------
.PHONY: train-local
train-local: ## Executa treino otimizado localmente (fora do Docker)
	$(PYTHON) -m src.models.train_fraud_model

.PHONY: optimize-local
optimize-local: ## Otimização de hiperparâmetros local
	$(PYTHON) -m src.models.optimization --data $(FEATURED_DATASET) --config $(CONFIG_FILE)

.PHONY: validate-local
validate-local: ## Validação local sem Docker
	$(PYTHON) -m src.models.validation --data $(FEATURED_DATASET) --model $(MODEL_ARTIFACT) --reference $(PRIMARY_DATASET)

.PHONY: interpret-local
interpret-local: ## Interpretação local
	$(PYTHON) -m src.models.interpretation --model $(MODEL_ARTIFACT) --data $(FEATURED_DATASET)

.PHONY: docs
docs: ## Abre documentação relevante no navegador (se disponível)
	@echo "API Docs: $(HOST_API_URL)/docs"
	@echo "MLflow UI: $(HOST_MLFLOW_URL)"
	@echo "Dashboard: $(HOST_DASH_URL)"
