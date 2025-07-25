# Makefile - TrustShield Advanced
.PHONY: help install test lint format clean docker-build docker-run

# === HELP ===
help:
	@echo "TrustShield Advanced - Comandos Disponíveis:"
	@echo "  install      - Instalar dependências"
	@echo "  test         - Executar testes"
	@echo "  lint         - Verificar código"
	@echo "  format       - Formatar código"
	@echo "  train        - Treinar todos os modelos com MLflow"
	@echo "  train-single - Treinar apenas o Isolation Forest"
	@echo "  mlflow       - Iniciar MLflow UI"
	@echo "  docker-build - Construir a imagem Docker"
	@echo "  docker-run   - Executar os serviços em Docker"
	@echo "  clean        - Limpar arquivos temporários"

# === DEVELOPMENT ===
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

# === TRAINING & MLFLOW ===
train:
	python src/models/train_fraud_model.py --config config/config.yaml --model all

train-single:
	python src/models/train_fraud_model.py --config config/config.yaml --model isolation_forest

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

# === DOCKER ===
docker-build:
	docker build -t trustshield-advanced:latest -f docker/Dockerfile .

docker-run:
	docker-compose -f docker/docker-compose.yml up -d

docker-stop:
	docker-compose -f docker/docker-compose.yml down

# === CLEANUP ===
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/
