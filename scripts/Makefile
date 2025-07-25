# Makefile - TrustShield Advanced
.PHONY: help install test lint format clean docker-build docker-run

# === HELP ===
help:
	@echo "TrustShield Advanced - Comandos Disponíveis:"
	@echo "  install      - Instalar dependências"
	@echo "  test         - Executar testes"
	@echo "  lint         - Verificar código"
	@echo "  format       - Formatar código"
	@echo "  train        - Treinar modelos"
	@echo "  mlflow       - Iniciar MLflow UI"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Executar em Docker"
	@echo "  clean        - Limpar arquivos temporários"

# === DEVELOPMENT ===
install:
	pip install -r requirements.txt
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

# === TRAINING ===
train:
	python train_advanced.py --config config/config_advanced.yaml --model all

train-single:
	python train_advanced.py --config config/config_advanced.yaml --model isolation_forest

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

# === DOCKER ===
docker-build:
	docker build -t trustshield-advanced:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

# === CLEANUP ===
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/

# === MONITORING ===
monitor:
	python -c "import psutil; print(f'CPU: {psutil.cpu_percent()}% | RAM: {psutil.virtual_memory().percent}%')"

# === DATA ===
download-data:
	@echo "Implementar download de dados se necessário"

prepare-data:
	python src/data/prepare_dataset.py

# === DEPLOYMENT ===
deploy-local:
	docker-compose -f docker-compose.yml -f docker-compose.local.yml up -d

deploy-prod:
	@echo "Implementar deploy para produção"
