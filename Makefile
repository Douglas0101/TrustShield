# =========================
# TrustShield - Makefile (infra + treino/retrain)
# =========================
# Uso rápido:
#   make up                # sobe infra (Postgres/MinIO/MLflow/API/Dashboard) + health
#   make pipeline          # dataset -> features -> train -> evaluate -> promote
#   make retrain           # dataset -> features -> optimize -> train -> evaluate -> promote
#   make train             # apenas treino (não-interativo)
#   make optimize          # apenas HPO
#   make evaluate          # avalia modelos recentes
#   make promote           # promove último isolation_forest_* como default_model.joblib
#   make nuke CONFIRM=1    # limpeza radical do projeto
#   make logs-mlflow       # logs de um serviço específico (idem -api, -dashboard, -postgres, -minio)
#
# Parâmetros:
#   make train EXPERIMENT=TrustShield RUN_TAG=manual
#   make retrain EXPERIMENT=TrustShield HPO_TRIALS=50 RUN_TAG=retrain-2025-08
#
# Observação: todos os comandos de ML/ETL rodam DENTRO do serviço "api" via docker compose exec.

SHELL := /usr/bin/env bash -eo pipefail

# ---------- Projeto / Compose ----------
PROJECT ?= trustshield
export COMPOSE_PROJECT_NAME := $(PROJECT)
COMPOSE_FILE := docker/docker-compose.yml
COMPOSE      := docker compose -f $(COMPOSE_FILE)

# ---------- Parametrização de Treino ----------
EXPERIMENT ?= TrustShield
RUN_TAG    ?= ad-hoc
HPO_TRIALS ?= 30         # nº padrão de tentativas na otimização
PY         ?= python

# ---------- Secrets, portas e helpers ----------
SECRETS := secrets/minio_root_user.txt secrets/minio_root_password.txt secrets/postgres_password.txt
PORTS   := 5000 9000 9001 8000 8501

define check_file
	@if [[ ! -s "$(1)" ]]; then echo "❌ Arquivo obrigatório ausente/vazio: $(1)"; exit 1; else echo "✅ OK: $(1)"; fi
endef

define check_port
	@if command -v ss >/dev/null 2>&1; then \
	  (ss -ltn | grep -qE ":(?:$(1))\b") && echo "⚠️  Porta $(1) em uso" || echo "✅ Porta $(1) livre"; \
	elif command -v lsof >/dev/null 2>&1; then \
	  (lsof -i :$(1) -sTCP:LISTEN -P -n >/dev/null 2>&1) && echo "⚠️  Porta $(1) em uso" || echo "✅ Porta $(1) livre"; \
	else \
	  echo "ℹ️  Nem ss nem lsof disponíveis; pulando checagem da porta $(1)"; \
	fi
endef

# Espera um endpoint HTTP ficar OK (host)
define wait_url
	@bash -lc 'for i in {1..180}; do curl -fsS "$(1)" >/dev/null && exit 0; sleep 1; done; echo "⏳ Timeout: $(1)"; exit 1'
endef

# Executa um comando dentro do contêiner "api"
define exec_api
	$(COMPOSE) exec api sh -lc '$(1)'
endef

.PHONY: help
help: ## mostra este help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9\-_]+:.*##/ {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ---------- Validações ----------
.PHONY: check-env
check-env: ## verifica .env (sem variáveis aninhadas) e existência
	@if [[ ! -f ".env" ]]; then echo "⚠️  .env não encontrado (use .env.example -> .env)"; else echo "✅ .env encontrado"; fi
	@if [[ -f ".env" ]] && grep -E '\$\{[A-Za-z_][A-Za-z0-9_]*\}' .env >/dev/null; then \
		echo "❌ Seu .env contém variáveis aninhadas (\$\{VAR\}). Remova-as e deixe valores literais."; exit 1; \
	fi

.PHONY: check-secrets
check-secrets: ## verifica secrets obrigatórios
	@for f in $(SECRETS); do \
	  if [ ! -s "$$f" ]; then echo "❌ Arquivo obrigatório ausente/vazio: $$f"; exit 1; else echo "✅ OK: $$f"; fi; \
	done

.PHONY: doctor
doctor:
	@command -v docker >/dev/null || { echo "❌ Docker não encontrado"; exit 1; }
	@docker version >/dev/null && echo "✅ Docker OK"
	@docker compose version >/dev/null && echo "✅ Docker Compose OK"
	@command -v curl >/dev/null || { echo "❌ curl não encontrado (necessário p/ alguns checks)"; exit 1; }
	@$(MAKE) check-env
	@$(MAKE) check-secrets
	@python - <<'PY'
	import os, socket
	ports = [5000, int(os.environ.get("MINIO_PORT_API", "9000")), int(os.environ.get("MINIO_PORT_CONSOLE","9001")), 8000, 8501]
	def busy(p):
		s=socket.socket(); s.settimeout(0.25)
		try: s.connect(("127.0.0.1", p)); s.close(); return True
		except: return False
	for p in ports:
		print(("⚠️  Porta %d em uso" if busy(p) else "✅ Porta %d livre") % p)
	PY


# ---------- Compose ----------
.PHONY: config
config: ## mostra o docker compose já resolvido (bom pra diagnosticar variáveis)
	$(COMPOSE) config

.PHONY: up
up: doctor ## sobe tudo com build e valida saúde
	$(COMPOSE) up -d --build
	@$(call wait_url,http://localhost:5000/version)
	@$(call wait_url,http://localhost:8000/healthz)
	@$(MAKE) health

.PHONY: up-no-build
up-no-build: ## sobe sem rebuild (idempotente)
	$(COMPOSE) up -d
	@$(call wait_url,http://localhost:5000/version) || true
	@$(call wait_url,http://localhost:8000/healthz) || true

.PHONY: build
build: ## (re)build das imagens com pull de bases
	$(COMPOSE) build --pull

.PHONY: rebuild
rebuild: ## rebuild completo sem cache
	$(COMPOSE) build --no-cache --pull

.PHONY: pull
pull: ## atualiza imagens base
	$(COMPOSE) pull

.PHONY: restart
restart: ## reinicia serviços
	$(COMPOSE) restart

.PHONY: ps
ps: ## status
	$(COMPOSE) ps

.PHONY: logs
logs: ## logs de todos os serviços
	$(COMPOSE) logs -f --tail=200

.PHONY: logs-mlflow logs-api logs-dashboard logs-postgres logs-minio
logs-mlflow:    ; $(COMPOSE) logs -f --tail=200 mlflow
logs-api:       ; $(COMPOSE) logs -f --tail=200 api
logs-dashboard: ; $(COMPOSE) logs -f --tail=200 dashboard
logs-postgres:  ; $(COMPOSE) logs -f --tail=200 postgres
logs-minio:     ; $(COMPOSE) logs -f --tail=200 minio

.PHONY: down
down: ## derruba (mantém volumes)
	$(COMPOSE) down --remove-orphans

.PHONY: down-v
down-v: ## derruba e remove volumes
	$(COMPOSE) down -v --remove-orphans

# ---------- Limpeza pesada ----------
.PHONY: prune
prune: ## remove recursos órfãos globais
	@echo "⚠️  Isto removerá recursos DORMENTES globalmente."
	@read -p "Continuar? [y/N] " ans; \
	[[ $$ans == "y" || $$ans == "Y" ]] || exit 1
	docker system prune -f
	docker volume prune -f
	docker builder prune -f

.PHONY: nuke
nuke: ## limpeza radical do projeto (down -v + volumes/redes/imagens do projeto). use: make nuke CONFIRM=1
	@if [[ "$(CONFIRM)" != "1" ]]; then \
		echo "❌ Proteção ativa. Rode: make nuke CONFIRM=1"; exit 1; \
	fi
	$(COMPOSE) down -v --remove-orphans || true
	-docker volume rm -f $(PROJECT)_pgdata $(PROJECT)_minio_data 2>/dev/null || true
	-docker network rm $(PROJECT)_default 2>/dev/null || true
	-docker images --filter "label=com.docker.compose.project=$(PROJECT)" -q | xargs -r docker rmi -f
	@echo "Executando prune final (imagens/containers órfãos)..."
	docker system prune -f

# ---------- Saúde ----------
.PHONY: health
health: ## checa endpoints principais (host)
	@set -e; \
	curl -fsS http://localhost:5000/version >/dev/null && echo "✅ MLflow OK" || (echo "❌ MLflow falhou" && false); \
	curl -fsS http://localhost:9000/minio/health/ready >/dev/null && echo "✅ MinIO OK" || (echo "❌ MinIO falhou" && false); \
	curl -fsS http://localhost:8000/healthz >/dev/null && echo "✅ API OK" || (echo "❌ API falhou" && false); \
	curl -fsS http://localhost:8501/_stcore/health >/dev/null && echo "✅ Dashboard OK" || (echo "❌ Dashboard falhou" && false)

# ---------- Utilidades ----------
.PHONY: psql
psql: ## abre psql no Postgres do compose (usa secret)
	@$(COMPOSE) exec -e PGPASSWORD="$$(cat secrets/postgres_password.txt 2>/dev/null || echo mlflow)" postgres \
		sh -lc 'psql -U $$POSTGRES_USER -d $$POSTGRES_DB'

.PHONY: mc
mc: ## shell do MinIO Client autenticado (mesma rede do compose)
	@docker run --rm -it \
		--network $(PROJECT)_default \
		-e MINIO_ROOT_USER="$$(cat secrets/minio_root_user.txt)" \
		-e MINIO_ROOT_PASSWORD="$$(cat secrets/minio_root_password.txt)" \
		minio/mc sh -lc '\
		  mc alias set local http://minio:9000 "$$MINIO_ROOT_USER" "$$MINIO_ROOT_PASSWORD" && \
		  echo "✅ mc conectado (alias: local)" && mc alias list && sh'

# =======================================================
#                TREINAMENTO & RETRAINING
# =======================================================

.PHONY: ensure-up
ensure-up: ## sobe infra sem rebuild e espera MLflow/API ficarem OK
	@$(MAKE) up-no-build
	@$(call wait_url,http://localhost:5000/version) || true
	@$(call wait_url,http://localhost:8000/healthz) || true

.PHONY: experiment
experiment: ensure-up ## cria (se necessário) experimento no MLflow com nome $(EXPERIMENT)
	@$(call exec_api, mlflow experiments create --experiment-name "$(EXPERIMENT)" --artifact-location "s3://$${MLFLOW_S3_BUCKET:-mlflow}/$(EXPERIMENT)" || true)
	@echo "✅ Experimento pronto: $(EXPERIMENT)"

.PHONY: data
data: ensure-up ## ingestão/curadoria -> gera data/processed/primary_dataset.parquet
	@$(call exec_api, $(PY) -m src.data.make_dataset)

.PHONY: features
features: ensure-up ## feature engineering -> gera data/features/featured_dataset.parquet
	@$(call exec_api, $(PY) -m src.features.build_features)

.PHONY: train
train: ensure-up experiment ## treino otimizado (sem prompts) via IntelI3Optimizer
	@$(call exec_api, $(PY) -c "from src.models.train_fraud_model import IntelI3Optimizer; IntelI3Optimizer().retrain_all_models_optimized()")

.PHONY: optimize
optimize: ensure-up experiment ## HPO (Optuna) sobre featured_dataset.parquet
	@$(call exec_api, $(PY) -m src.models.optimization --data data/features/featured_dataset.parquet --methods optuna --trials $(HPO_TRIALS))

.PHONY: evaluate
evaluate: ensure-up
	@$(call exec_api, MODELS=$$(ls -1t outputs/models/isolation_forest_*.joblib 2>/dev/null | head -n 5 || echo outputs/models/default_model.joblib); echo "Avaliando: $$MODELS"; $(PY) -m src.models.evaluate_models --data data/features/featured_dataset.parquet --models $$MODELS)

.PHONY: promote stage
promote stage: ensure-up ## último isolation_forest_* → default_model.joblib (consumido pela API)
	@$(call exec_api, set -e; cd /app/outputs/models; LATEST=$$(ls -1t isolation_forest_*.joblib 2>/dev/null | head -n1); if [ -z "$$LATEST" ]; then echo "❌ Nenhum modelo isolation_forest_* encontrado"; exit 1; fi; cp "$$LATEST" default_model.joblib; echo "✅ Promovido: $$LATEST -> default_model.joblib")

.PHONY: pipeline
pipeline: data features train evaluate stage ## E2E: dados → features → treino → avaliação → promoção
	@echo "✅ Pipeline concluído (experimento: $(EXPERIMENT), tag: $(RUN_TAG))"

.PHONY: retrain
retrain: data features optimize train evaluate stage ## retraining com HPO + promoção
	@echo "✅ Retraining concluído (exp: $(EXPERIMENT), trials: $(HPO_TRIALS), tag: $(RUN_TAG))"

# ---------- Testes ----------
.PHONY: test
test: ## roda a suíte de testes (pytest) no host
	pytest -q

# target default
.DEFAULT_GOAL := help
