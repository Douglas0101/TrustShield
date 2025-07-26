# Makefile - TrustShield Advanced (Versão 5.1.0-stable)
.PHONY: help install test lint format clean docker-build docker-run docker-logs docker-stop

# === AJUDA ===
help:
	@echo "TrustShield Advanced - Comandos Disponíveis:"
	@echo "  install      - Instalar dependências"
	@echo "  test         - Executar testes"
	@echo "  lint         - Verificar o estilo do código"
	@echo "  format       - Formatar o código"
	@echo "  train        - Treinar todos os modelos com MLflow"
	@echo "  mlflow       - Iniciar a interface do MLflow localmente"
	@echo "  docker-build - Construir a imagem Docker da aplicação"
	@echo "  docker-run   - Executar todos os serviços Docker em segundo plano"
	@echo "  docker-logs  - Ver os logs do contentor de treino em tempo real"
	@echo "  docker-stop  - Parar e remover todos os serviços Docker"
	@echo "  clean        - Limpar ficheiros temporários do Python"

# === DESENVOLVIMENTO ===
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

# === TREINO & MLFLOW ===
train:
	python src/models/train_fraud_model.py --config config/config.yaml --model all

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

# === DOCKER ===
# Comandos atualizados para a sintaxe moderna do Docker Compose
docker-build:
	docker build -t trustshield-advanced:latest -f docker/Dockerfile .

docker-run:
	docker compose -f docker/docker-compose.yml up -d

docker-logs:
	docker compose -f docker/docker-compose.yml logs -f trustshield-trainer

docker-stop:
	docker compose -f docker/docker-compose.yml down --volumes

# === LIMPEZA ===
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/
