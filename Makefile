# Makefile — TrustShield (motor principal)
# Orquestra Docker Compose, dados e pipelines de IA
# Uso: `make help`

SHELL := /bin/bash
.ONESHELL:
.EXPORT_ALL_VARIABLES:

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
COMPOSE_FILE := docker/docker-compose.yml
COMPOSE      := docker compose -f $(COMPOSE_FILE)
SERVICES     := postgres minio mlflow api dashboard

API_HEALTH   := http://localhost:8000/healthz
DASH_HEALTH  := http://localhost:8501/_stcore/health
MLFLOW_HEALTH:= http://localhost:5000/

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Docker / Compose
# ---------------------------------------------------------------------------
up: ensure-dirs verify-secrets init-bucket ## Build e sobe todos os serviços (+build)
	$(COMPOSE) up -d --build

init-bucket: ## (Re)executa job para criar bucket 'mlflow' no MinIO
	$(COMPOSE) run --rm minio-mc-init

build: ## Build das imagens
	$(COMPOSE) build

rebuild: ## Rebuild sem cache + up
	$(COMPOSE) build --no-cache
	$(COMPOSE) up -d

ps: ## Status dos serviços
	$(COMPOSE) ps

logs: ## Logs de api, dashboard, mlflow
	$(COMPOSE) logs -f api dashboard mlflow

logs-%: ## Logs de um serviço. Ex: make logs-api
	$(COMPOSE) logs -f $*

restart: ## Reinicia api e dashboard
	$(COMPOSE) restart api dashboard

shell-%: ## Shell no container (bash/sh). Ex: make shell-api
	-$(COMPOSE) exec $* bash || $(COMPOSE) exec $* sh

down: ## Para e remove containers (mantém volumes)
	$(COMPOSE) down

clean: ## Para e remove containers + volumes
	$(COMPOSE) down -v

prune: ## Limpa imagens/volumes não usados
	docker system prune -f

orphan-clean: ## Remove órfãos (containers de versões antigas)
	$(COMPOSE) down --remove-orphans || true

# ---------------------------------------------------------------------------
# Saúde / Abertura
# ---------------------------------------------------------------------------
health: ## Checa API, Dashboard e MLflow
	@echo "API:"; curl -fsS $(API_HEALTH) && echo || exit 1
	@echo "Dashboard:"; curl -fsS $(DASH_HEALTH) && echo || curl -fsS http://localhost:8501/ >/dev/null || exit 1
	@echo "MLflow:"; curl -fsS $(MLFLOW_HEALTH) >/dev/null && echo "OK" || exit 1

open: ## Abre UIs
	@if command -v xdg-open >/dev/null; then \
	  xdg-open http://localhost:8501; \
	  xdg-open http://localhost:8000/docs; \
	  xdg-open http://localhost:5000; \
	fi

wait-api: ## Aguarda API saudável (timeout 120s)
	@echo "Aguardando API em $(API_HEALTH) ..."; \
	for i in {1..60}; do \
	  if curl -fsS $(API_HEALTH) >/dev/null; then echo "API OK"; exit 0; fi; \
	  sleep 2; \
	done; echo "Timeout aguardando API"; exit 1

# ---------------------------------------------------------------------------
# Dev local (sem Docker)
# ---------------------------------------------------------------------------
api-local: ## Sobe API local
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dash-local: ## Sobe Dashboard local apontando p/ API local
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
	API_URL=http://127.0.0.1:8000 \
	streamlit run src/dashboard/app.py --server.headless=true --server.fileWatcherType=none --server.runOnSave=false --logger.level=error --server.port=8501 --server.address=0.0.0.0

# ---------------------------------------------------------------------------
# Pipelines (rodam dentro do container da API)
# ---------------------------------------------------------------------------
make-features: ## Gera features
	$(COMPOSE) exec api python -m src.data.make_dataset

train: ## Treina modelo padrão
	$(COMPOSE) exec api python -m src.models.train_fraud_model

retrain30: ## Re-treina os 30 modelos otimizados com logging no MLflow
	$(COMPOSE) exec api python -c "from src.models.optimization import IntelI3Optimizer; IntelI3Optimizer().retrain_all_models_optimized()"

validate: ## Validação / detecção de drift
	$(COMPOSE) exec api python -c "from src.models.validation import ResilientTrustShieldValidator as V; v=V(); v.run_validation(data_path='data/interim/validation_sample.parquet', model_path='outputs/models/default_model.joblib', reference_data_path='data/features/featured_dataset.parquet', validation_types=['drift_detection']); print('Validation finished')"

# Exemplo de avaliação (ajuste modelos conforme necessário)
eval: ## Avalia modelos e gera relatório (ajuste lista de modelos)
	$(COMPOSE) exec api python -m src.models.evaluate_models --data data/features/featured_dataset.parquet --types performance --config config/config.yaml || true

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
ensure-dirs: ## Cria diretórios de runtime
	mkdir -p data outputs outputs/validation outputs/interpretations outputs/temp_explanations

verify-secrets: ## Verifica secrets necessários
	@test -f secrets/minio_root_user.txt || (echo "Falta secrets/minio_root_user.txt" && exit 1)
	@test -f secrets/minio_root_password.txt || (echo "Falta secrets/minio_root_password.txt" && exit 1)
	@test -f secrets/postgres_password.txt || (echo "Falta secrets/postgres_password.txt" && exit 1)

# ---------------------------------------------------------------------------
# Diagnóstico
# ---------------------------------------------------------------------------
doctor: ## Diagnóstico do stack (config, saúde e conectividade)
	@echo "== Compose config (trechos principais) =="; \
	$(COMPOSE) config | sed -n '1,120p'; \
	echo; echo "== Serviços =="; $(COMPOSE) ps; \
	echo; echo "== Saúde =="; \
	curl -fsS $(MLFLOW_HEALTH) >/dev/null && echo "MLflow OK" || echo "MLflow NOK"; \
	curl -fsS $(API_HEALTH)     >/dev/null && echo "API OK"    || echo "API NOK"; \
	curl -fsS $(DASH_HEALTH)    >/dev/null && echo "Dash OK"   || echo "Dash NOK";

# ---------------------------------------------------------------------------
# Ajuda
# ---------------------------------------------------------------------------
help: ## Mostra esta ajuda
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## ' $(MAKEFILE_LIST) | sed -e 's/:.*## /: /' | sort
# Makefile — TrustShield (motor principal)
# Orquestra Docker Compose, dados e pipelines de IA
# Uso: `make help`

SHELL := /bin/bash
.ONESHELL:
.EXPORT_ALL_VARIABLES:

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
COMPOSE_FILE := docker/docker-compose.yml
COMPOSE      := docker compose -f $(COMPOSE_FILE)
SERVICES     := postgres minio mlflow api dashboard

API_HEALTH   := http://localhost:8000/healthz
DASH_HEALTH  := http://localhost:8501/_stcore/health
MLFLOW_HEALTH:= http://localhost:5000/

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Docker / Compose
# ---------------------------------------------------------------------------
up: ensure-dirs verify-secrets ## Build e sobe todos os serviços
	$(COMPOSE) up -d --build

build: ## Build das imagens
	$(COMPOSE) build

rebuild: ## Rebuild sem cache + up
	$(COMPOSE) build --no-cache
	$(COMPOSE) up -d

ps: ## Status dos serviços
	$(COMPOSE) ps

logs: ## Logs de api, dashboard, mlflow
	$(COMPOSE) logs -f api dashboard mlflow

logs-%: ## Logs de um serviço. Ex: make logs-api
	$(COMPOSE) logs -f $*

restart: ## Reinicia api e dashboard
	$(COMPOSE) restart api dashboard

shell-%: ## Shell no container (bash/sh). Ex: make shell-api
	-$(COMPOSE) exec $* bash || $(COMPOSE) exec $* sh

down: ## Para e remove containers (mantém volumes)
	$(COMPOSE) down

clean: ## Para e remove containers + volumes
	$(COMPOSE) down -v

prune: ## Limpa imagens/volumes não usados
	docker system prune -f

orphan-clean: ## Remove órfãos (containers de versões antigas)
	$(COMPOSE) down --remove-orphans || true

# ---------------------------------------------------------------------------
# Saúde / Abertura
# ---------------------------------------------------------------------------
health: ## Checa API, Dashboard e MLflow
	@echo "API:"; curl -fsS $(API_HEALTH) && echo || exit 1
	@echo "Dashboard:"; curl -fsS $(DASH_HEALTH) && echo || curl -fsS http://localhost:8501/ >/dev/null || exit 1
	@echo "MLflow:"; curl -fsS $(MLFLOW_HEALTH) >/dev/null && echo "OK" || exit 1

open: ## Abre UIs
	@if command -v xdg-open >/dev/null; then \
	  xdg-open http://localhost:8501; \
	  xdg-open http://localhost:8000/docs; \
	  xdg-open http://localhost:5000; \
	fi

wait-api: ## Aguarda API saudável (timeout 120s)
	@echo "Aguardando API em $(API_HEALTH) ..."; \
	for i in {1..60}; do \
	  if curl -fsS $(API_HEALTH) >/dev/null; then echo "API OK"; exit 0; fi; \
	  sleep 2; \
	done; echo "Timeout aguardando API"; exit 1

# ---------------------------------------------------------------------------
# Dev local (sem Docker)
# ---------------------------------------------------------------------------
api-local: ## Sobe API local
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dash-local: ## Sobe Dashboard local apontando p/ API local
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
	API_URL=http://127.0.0.1:8000 \
	streamlit run src/dashboard/app.py --server.headless=true --server.fileWatcherType=none --server.runOnSave=false --logger.level=error --server.port=8501 --server.address=0.0.0.0

# ---------------------------------------------------------------------------
# Pipelines (rodam dentro do container da API)
# ---------------------------------------------------------------------------
make-features: ## Gera features
	$(COMPOSE) exec api python -m src.data.make_dataset

train: ## Treina modelo padrão
	$(COMPOSE) exec api python -m src.models.train_fraud_model

retrain30: ## Re-treina os 30 modelos otimizados com logging no MLflow
	$(COMPOSE) exec api python -c "from src.models.optimization import IntelI3Optimizer; IntelI3Optimizer().retrain_all_models_optimized()"

validate: ## Validação / detecção de drift
	$(COMPOSE) exec api python -c "from src.models.validation import ResilientTrustShieldValidator as V; v=V(); v.run_validation(data_path='data/interim/validation_sample.parquet', model_path='outputs/models/default_model.joblib', reference_data_path='data/features/featured_dataset.parquet', validation_types=['drift_detection']); print('Validation finished')"

# Exemplo de avaliação (ajuste modelos conforme necessário)
eval: ## Avalia modelos e gera relatório (ajuste lista de modelos)
	$(COMPOSE) exec api python -m src.models.evaluate_models --data data/features/featured_dataset.parquet --types performance --config config/config.yaml || true

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
ensure-dirs: ## Cria diretórios de runtime
	mkdir -p data outputs outputs/validation outputs/interpretations outputs/temp_explanations

verify-secrets: ## Verifica secrets necessários
	@test -f secrets/minio_root_user.txt || (echo "Falta secrets/minio_root_user.txt" && exit 1)
	@test -f secrets/minio_root_password.txt || (echo "Falta secrets/minio_root_password.txt" && exit 1)
	@test -f secrets/postgres_password.txt || (echo "Falta secrets/postgres_password.txt" && exit 1)

# ---------------------------------------------------------------------------
# Diagnóstico
# ---------------------------------------------------------------------------
doctor: ## Diagnóstico do stack (config, saúde e conectividade)
	@echo "== Compose config (trechos principais) =="; \
	$(COMPOSE) config | sed -n '1,120p'; \
	echo; echo "== Serviços =="; $(COMPOSE) ps; \
	echo; echo "== Saúde =="; \
	curl -fsS $(MLFLOW_HEALTH) >/dev/null && echo "MLflow OK" || echo "MLflow NOK"; \
	curl -fsS $(API_HEALTH)     >/dev/null && echo "API OK"    || echo "API NOK"; \
	curl -fsS $(DASH_HEALTH)    >/dev/null && echo "Dash OK"   || echo "Dash NOK";

# ---------------------------------------------------------------------------
# Ajuda
# ---------------------------------------------------------------------------
help: ## Mostra esta ajuda
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## ' $(MAKEFILE_LIST) | sed -e 's/:.*## /: /' | sort
