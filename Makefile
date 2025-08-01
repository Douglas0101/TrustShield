# Makefile - TrustShield Advanced (Versão 9.0.0 - Robusto e Otimizado)
.PHONY: help install test lint format clean services-up services-down services-up-fresh train logs purge

# --- CONFIGURAÇÃO E PRÉ-REQUISITOS ---

# Verifica a existência do arquivo .env, que é crucial para a configuração.
# Se não existir, interrompe a execução com uma mensagem de ajuda.
ifeq ($(wildcard ./.env),)
    $(error .env file not found. Please create it by copying .env.example: `cp .env.example .env`)
endif

# Define o comando base do Docker Compose para ser usado em todo o Makefile.
DOCKER_COMPOSE_CMD = docker compose --env-file ./.env

# =====================================================================================
# === SEÇÃO DOCKER: O CORAÇÃO DA OPERAÇÃO ===
# =====================================================================================

services-up:
	@echo "🚀 Subindo todos os serviços de suporte e aguardando que fiquem saudáveis..."
	# O flag --wait é crucial: ele pausa até que os healthchecks de todos os serviços passem.
	$(DOCKER_COMPOSE_CMD) up -d --wait
	@echo "✅ Todos os serviços estão prontos e saudáveis."

services-down:
	@echo "🛑 Parando todos os serviços..."
	# --remove-orphans limpa contêineres que não são mais necessários.
	$(DOCKER_COMPOSE_CMD) down --remove-orphans

services-up-fresh: services-down
	@echo "🧼 Reconstruindo imagens e subindo todos os serviços do zero..."
	# Adicionado --wait para consistência e robustez. Garante que mesmo após
	# uma reconstrução, o comando só termina quando tudo está saudável.
	$(DOCKER_COMPOSE_CMD) up -d --build --force-recreate --remove-orphans --wait
	@echo "✅ Todos os serviços foram reconstruídos e estão prontos e saudáveis."

# Define 'mlflow' como o serviço padrão para logs, mas permite override.
# Exemplo: make logs service=trustshield-api
service ?= mlflow
logs:
	@echo "🔎 Acompanhando os logs do serviço: $(service)... (Pressione Ctrl+C para sair)"
	$(DOCKER_COMPOSE_CMD) logs -f $(service)

# Define o modelo padrão para treino, mas permite override.
# Exemplo: make train args="--model random_forest"
args ?= --model isolation_forest
process-data:
	@echo "🔧 Executando o pipeline de processamento de dados..."
	$(DOCKER_COMPOSE_CMD) exec trustshield-api python /home/trustshield/src/data/make_dataset.py
	$(DOCKER_COMPOSE_CMD) exec trustshield-api python /home/trustshield/src/features/build_features.py
	@echo "✅ Processamento de dados concluído."

train: services-up process-data # <-- REVISÃO CRÍTICA: Adiciona dependência para garantir que tudo está UP.
	@echo "🧠 Ambiente pronto. Executando o pipeline de treino no contêiner da API..."
	@echo "   Comando: python /home/trustshield/src/models/train_fraud_model.py $(args)"
	$(DOCKER_COMPOSE_CMD) exec trustshield-api python /home/trustshield/src/models/train_fraud_model.py $(args)
	@echo "✅ Treino concluído."

purge:
	@echo "🔥🔥🔥 ATENÇÃO: Parando e APAGANDO TODOS OS DADOS! 🔥🔥🔥"
	$(DOCKER_COMPOSE_CMD) down --volumes
	@echo "🧹 Limpando recursos do Docker (cache de build e imagens não utilizadas)..."
	docker builder prune -a -f
	docker system prune -f
	@echo "🧼 Limpeza completa."

# =====================================================================================
# === SEÇÃO DE DESENVOLVIMENTO LOCAL ===
# =====================================================================================

help:
	@echo "TrustShield Advanced - Comandos Disponíveis:"
	@echo ""
	@echo "--- GESTÃO DE SERVIÇOS ---"
	@echo "  services-up         - Inicia todos os serviços e aguarda ficarem saudáveis."
	@echo "  services-down       - Para todos os serviços."
	@echo "  services-up-fresh   - PARA, reconstrói e reinicia todos os serviços, aguardando ficarem saudáveis."
	@echo "  logs [service=...]  - Mostra os logs de um serviço (padrão: mlflow)."
	@echo "  purge               - PARA TUDO e APAGA TODOS os dados (contêineres, volumes, redes)."
	@echo ""
	@echo "--- PIPELINE & TAREFAS ---"
	@echo "  train [args=...]    - Executa o pipeline de treino (padrão: --model isolation_forest)."
	@echo ""
	@echo "--- QUALIDADE DE CÓDIGO ---"
	@echo "  install             - Instala dependências locais e hooks de pré-commit."
	@echo "  test                - Roda os testes com pytest."
	@echo "  lint                - Roda os linters (flake8, mypy)."
	@echo "  format              - Formata o código com black."
	@echo "  clean               - Remove arquivos temporários do Python."


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