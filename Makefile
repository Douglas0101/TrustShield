# ===============================
# TrustShield — Makefile (Docker) V2
# ===============================
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

PROJECT        ?= trustshield
COMPOSE_FILE   ?= docker/docker-compose.yml
DC             := DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker compose -p $(PROJECT) -f $(COMPOSE_FILE)

API_SERVICE    ?= trustshield-api
DASH_SERVICE   ?= trustshield-dashboard
MLFLOW_SERVICE ?= mlflow
MINIO_SERVICE  ?= minio

# --- CORREÇÃO APLICADA AQUI ---
# A URL da API agora usa uma variável de ambiente, com 8080 como padrão.
# Para usar outra porta, execute: export API_HOST_PORT=8001 && make up
API_HOST_PORT  ?= 8080
HOST_API_URL   ?= http://localhost:$(API_HOST_PORT)
HOST_DASH_URL  ?= http://localhost:8501
HOST_MLFLOW_URL?= http://localhost:5500

NO_CACHE ?= 0
PULL     ?= 0
BUILD_ARGS :=
ifeq ($(NO_CACHE),1)
  BUILD_ARGS += --no-cache
endif
ifeq ($(PULL),1)
  BUILD_ARGS += --pull always
endif

.PHONY: help
help:
	@echo "make up|down|restart|build|rebuild|logs|api-logs|dash-logs|ps|top|wait|health|clean|prune|env"
	@echo "Treino: make data features train eval optimize validate interpret promote reload smoke cycle quick"

.PHONY: up down restart build rebuild logs api-logs dash-logs ps top wait health check clean prune env \
        api-shell dash-shell mlflow-shell minio-shell \
        data features train eval optimize validate interpret promote reload smoke cycle quick

up: check
	$(DC) build $(BUILD_ARGS)
	$(DC) up -d
	@$(MAKE) wait

down:
	$(DC) down --remove-orphans

restart: down up
build: check ; $(DC) build $(BUILD_ARGS)

rebuild:
	@$(MAKE) build NO_CACHE=1
	$(DC) up -d
	@$(MAKE) wait

logs: ; $(DC) logs -f
api-logs: ; $(DC) logs -f $(API_SERVICE)
dash-logs: ; $(DC) logs -f $(DASH_SERVICE)
ps: ; $(DC) ps
top: ; $(DC) top

wait:
	@echo "Aguardando saúde da API ($(API_SERVICE))..."
	@cid=$$($(DC) ps -q $(API_SERVICE)); \
	[ -n "$$cid" ] || { echo "ERRO: container da API não encontrado."; exit 1; }; \
	for i in $$(seq 1 60); do \
	  s=$$(docker inspect -f '{{.State.Health.Status}}' $$cid 2>/dev/null || echo "unknown"); \
	  echo "  tentativa $$i: $$s"; \
	  [ "$$s" = "healthy" ] && break; \
	  sleep 2; \
	done; \
	[ "$$s" = "healthy" ] || { echo "ERRO: API não ficou healthy a tempo."; exit 1; }; \
	curl -fsS $(HOST_API_URL)/healthz >/dev/null && echo "OK: /healthz"

health:
	@cid=$$($(DC) ps -q $(API_SERVICE)); \
	[ -n "$$cid" ] || { echo "ERRO: container da API não encontrado."; exit 1; }; \
	docker inspect -f 'Status={{.State.Status}} Health={{if .State.Health}}{{.State.Health.Status}}{{else}}(sem healthcheck){{end}}' $$cid ; \
	echo; curl -i --max-time 3 $(HOST_API_URL)/healthz || true

api-shell: ; @cid=$$($(DC) ps -q $(API_SERVICE));  [ -n "$$cid" ] && (docker exec -it $$cid bash || docker exec -it $$cid sh)
dash-shell: ; @cid=$$($(DC) ps -q $(DASH_SERVICE)); [ -n "$$cid" ] && (docker exec -it $$cid bash || docker exec -it $$cid sh)
mlflow-shell: ; @cid=$$($(DC) ps -q $(MLFLOW_SERVICE));[ -n "$$cid" ] && (docker exec -it $$cid bash || docker exec -it $$cid sh)
minio-shell: ; @cid=$$($(DC) ps -q $(MINIO_SERVICE)); [ -n "$$cid" ] && docker exec -it $$cid sh

check:
	@command -v docker >/dev/null 2>&1 || { echo "ERRO: docker não encontrado"; exit 1; }
	@docker compose version >/dev/null 2>&1 || { echo "ERRO: docker compose v2 não encontrado"; exit 1; }
	@[ -f $(COMPOSE_FILE) ] || { echo "ERRO: $(COMPOSE_FILE) não existe"; exit 1; }
	@$(DC) config >/dev/null || { echo "ERRO: 'docker compose config' falhou"; exit 1; }
	@grep -q '$(API_SERVICE):' $(COMPOSE_FILE)  || { echo "ERRO: serviço $(API_SERVICE) ausente"; exit 1; }
	@grep -q '$(DASH_SERVICE):' $(COMPOSE_FILE) || { echo "ERRO: serviço $(DASH_SERVICE) ausente"; exit 1; }
	@echo "OK: pré-checagens passaram."

clean: down
	@vols=$$(docker volume ls -q | grep -E '^$(PROJECT)_' || true); \
	[ -n "$$vols" ] && docker volume rm $$vols || echo "Sem volumes nomeados para remover."

prune:
	@echo "ATENÇÃO: limpa imagens/containers/volumes não usados (host inteiro)"; docker system prune -af --volumes

env:
	@echo "PROJECT=$(PROJECT) COMPOSE_FILE=$(COMPOSE_FILE)"
	@echo "API=$(API_SERVICE) DASH=$(DASH_SERVICE) MLFLOW=$(MLFLOW_SERVICE) MINIO=$(MINIO_SERVICE)"
	@echo "HOSTS: API=$(HOST_API_URL) DASH=$(HOST_DASH_URL) MLFLOW=$(HOST_MLFLOW_URL)"
	@echo "BUILD_ARGS='$(BUILD_ARGS)'"

# ---- pipeline de treino ----
data:      ; $(DC) exec $(API_SERVICE) python -m src.data.make_dataset
features:  ; $(DC) exec $(API_SERVICE) python -m src.features.build_features
train:     ; $(DC) exec $(API_SERVICE) python -m src.models.train_fraud_model
eval:      ; $(DC) exec $(API_SERVICE) python -m src.models.evaluate_models
optimize:  ; $(DC) exec $(API_SERVICE) python -m src.models.optimization
validate:  ; $(DC) exec $(API_SERVICE) python -m src.models.validation
interpret: ; $(DC) exec $(API_SERVICE) python -m src.models.interpretation
promote:   ; $(DC) exec $(API_SERVICE) python -c "from pathlib import Path; import shutil,sys; p=Path('outputs/models'); c=sorted(p.glob('isolation_forest_optimized_*.joblib'), key=lambda x: x.stat().st_mtime, reverse=True); sys.exit(0) if not c else (shutil.copy2(c[0], p/'default_model.joblib') or print('Promovido:', c[0].name))"
reload:    ; $(DC) restart $(API_SERVICE)
smoke:     ; curl -fsS $(HOST_API_URL)/healthz >/dev/null && echo "OK: /healthz" && curl -fsS $(HOST_API_URL)/status >/dev/null && echo "OK: /status"

cycle: data features train eval optimize promote reload wait smoke
quick: clean up cycle
