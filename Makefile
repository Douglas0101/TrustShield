# ==============================================================================
# Makefile - TrustShield Enterprise Grade
# Versão: 9.0.0 (Robust Build Flow)
#
# Otimizações e Melhores Práticas Implementadas:
# - ROBUSTEZ: O comando 'fresh' agora depende do 'purge', garantindo uma
#   limpeza completa antes de cada reconstrução para evitar conflitos.
# - CLAREZA: Comandos simplificados e ajuda detalhada.
# - MODERNIZAÇÃO: Uso exclusivo de 'docker compose' (sintaxe V2).
# ==============================================================================

# Define o nome do arquivo compose para não repetir.
COMPOSE_FILE := docker/docker-compose.yml

# Evita que o make confunda um alvo com um nome de arquivo.
.PHONY: help up down fresh logs train purge

# --- ALVO PADRÃO ---
# Executado quando 'make' é chamado sem argumentos.
default: help

# === AJUDA ===
help:
	@echo "=============== TrustShield MLOps Control Panel ================"
	@echo "Uso: make [comando]"
	@echo ""
	@echo "--- Gestão do Ambiente Docker ---"
	@echo "  up                  - Inicia todos os serviços em background."
	@echo "  down                - Para todos os serviços (sem apagar dados)."
	@echo "  fresh               - (RECOMENDADO) Limpa TUDO e reconstrói o ambiente do zero."
	@echo "  logs [service=...]  - Mostra os logs de um serviço (padrão: trustshield-api)."
	@echo ""
	@echo "--- Pipeline de Machine Learning ---"
	@echo "  train [args=...]    - Executa o pipeline de treino completo (ex: make train args='--config config/alternative.yaml')."
	@echo ""
	@echo "--- Limpeza Completa (AÇÃO DESTRUTIVA) ---"
	@echo "  purge               - PARA e APAGA todos os contêineres, redes e VOLUMES DE DADOS."

# ==============================================================================
# === Gestão do Ambiente Docker
# ==============================================================================
up:
	@echo "🚀 Iniciando todos os serviços do TrustShield em background..."
	docker compose -f $(COMPOSE_FILE) up -d

down:
	@echo "🛑 Parando todos os serviços do TrustShield..."
	docker compose -f $(COMPOSE_FILE) down

# OTIMIZAÇÃO: Este comando agora executa 'purge' primeiro, garantindo um ambiente limpo.
fresh: purge
	@echo "🔄 Reconstruindo imagens e reiniciando todos os serviços..."
	docker compose -f $(COMPOSE_FILE) up -d --build --force-recreate

# Permite especificar o serviço para os logs, ex: make logs service=mlflow
service ?= trustshield-api
logs:
	@echo "🔎 Acompanhando logs do serviço: $(service)... (Pressione Ctrl+C para sair)"
	docker compose -f $(COMPOSE_FILE) logs -f $(service)

# ==============================================================================
# === Pipeline de Machine Learning
# ==============================================================================

# Permite passar argumentos para o script, ex: make train args="--config config/other.yaml"
args ?= --config config/config.yaml
train: up
	@echo "🧠 Executando o pipeline de treino do TrustShield..."
	@echo "   Comando a ser executado no container:"
	@echo "   python src/models/train_fraud_model.py $(args)"
	# Usa 'run --rm' para criar um container efêmero para a tarefa de treino.
	docker compose -f $(COMPOSE_FILE) run --rm trustshield-api python src/models/train_fraud_model.py $(args)

# ==============================================================================
# === Limpeza Completa
# ==============================================================================
purge:
	@echo "🔥🔥🔥 AVISO: Ação destrutiva! Parando e apagando todos os contêineres, redes e volumes... 🔥🔥🔥"
	@echo "--> Forçando a parada e remoção de contêineres conhecidos para evitar conflitos..."
	@-docker stop trustshield-api trustshield-mlflow trustshield-bucket-creator trustshield-minio trustshield-postgres >/dev/null 2>&1
	@-docker rm -f trustshield-api trustshield-mlflow trustshield-bucket-creator trustshield-minio trustshield-postgres >/dev/null 2>&1
	@echo "--> Executando o 'down' do compose para limpar a rede e os volumes..."
	docker compose -f $(COMPOSE_FILE) down --volumes
	@echo "🧹 Limpando cache de build e outros recursos não utilizados do Docker..."
	docker builder prune -a -f
	docker system prune -f
	@echo "✨ Ambiente limpo."
