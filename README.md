# TrustShield - Advanced Fraud Detection System

**TrustShield** is an enterprise-grade, AI-powered fraud detection and prevention system designed for high-performance, real-time transaction processing. It leverages a sophisticated architecture and advanced machine learning models to identify and flag anomalous activities with high accuracy.

## 🚀 Key Features

- **Advanced AI Models**: Utilizes a suite of unsupervised learning models, including Isolation Forest, Local Outlier Factor (LOF), and One-Class SVM, to detect complex fraud patterns.
- **High-Performance Architecture**: Built on a hexagonal, domain-driven design (DDD) for scalability and maintainability, with a focus on high-throughput, low-latency inference.
- **MLflow Integration**: End-to-end experiment tracking, model versioning, and artifact storage powered by MLflow, ensuring reproducibility and auditability.
- **Dockerized Environment**: Fully containerized with Docker and Docker Compose for consistent, cross-platform deployments and simplified dependency management.
- **CI/CD Ready**: Includes a robust `Makefile` with commands for testing, linting, formatting, and building, ready for integration into any CI/CD pipeline.
- **Intel Optimized**: Performance-tuned for Intel architectures, maximizing CPU and memory efficiency for both training and inference.

## ⚙️ Project Structure

The project follows a clean, modular architecture:

```
TrustShield/
├── config/               # Configuration files (YAML)
├── data/                 # Raw, processed, and featured data
├── docker/               # Dockerfile and Docker Compose setup
├── logs/                 # Application and system logs
├── mlruns/               # MLflow experiment tracking data
├── notebooks/            # Jupyter notebooks for EDA
├── outputs/              # Trained models, reports, and figures
├── scripts/              # Automation scripts (Makefile)
├── src/                  # Core source code
│   ├── data/             # Data processing and loading
│   ├── features/         # Feature engineering
│   ├── models/           # Model training and prediction
│   └── utils/            # Utility functions
└── tests/                # Unit and integration tests
```

## 🏁 Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- `make`

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-repo/TrustShield.git
cd TrustShield
make install
```

### 2. Running with Docker

The recommended way to run the system is with Docker Compose, which orchestrates the training, MLflow, and database services.

**Build the Docker image:**

```bash
make docker-build
```

**Start all services:**

```bash
make docker-run
```

This will start:
- **`trustshield-trainer`**: The main application for training models.
- **`trustshield-mlflow`**: The MLflow tracking server.
- **`trustshield-postgres`**: The PostgreSQL database for MLflow.

You can access the MLflow UI at [http://localhost:5000](http://localhost:5000).

### 3. Training Models

To manually trigger a training run:

```bash
make train
```

This will train all configured models and log the results to MLflow. To train a specific model, use the `--model` flag:

```bash
python src/models/train_fraud_model.py --config config/config.yaml --model isolation_forest
```

## 🧪 Testing

To run the full test suite:

```bash
make test
```

## 🤝 Contributing

Contributions are welcome! Please refer to the project's development guidelines and submit a pull request.