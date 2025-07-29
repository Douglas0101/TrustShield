# -*- coding: utf-8 -*-
"""
Sistema de Treinamento Ultra-Avançado - Projeto TrustShield
VERSÃO EMPRESARIAL REARQUITETADA COM PADRÕES DE DESIGN CLÁSSICOS

🏆 APRIMORAMENTOS PROFUNDOS (v7.0.0-enhanced):
✅ Validação de Configuração: Schema validation com jsonschema para evitar erros de config.
✅ Otimização com Dask: Suporte a datasets grandes para escalabilidade.
✅ Hyperparameter Tuning: Integração com Optuna para tuning automático.
✅ Novo Modelo: Adicionado One-Class SVM.
✅ Error Handling Avançado: Retries, rollbacks e logs detalhados.
✅ Observabilidade: Mais logs no MLflow (artefatos, métricas de sistema).
✅ Performance: Caching de dados e garbage collection otimizada.

Autor: TrustShield Team - Enterprise Enhanced Version
Versão: 7.0.0-enhanced
Data: 2025-07-28
"""

import argparse
import gc
import json
import logging
import os
import psutil
import signal
import sys
import time
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Union, Tuple, Protocol, runtime_checkable

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from jsonschema import validate, ValidationError
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import optuna
from optuna.integration import MLflowCallback
import dask.dataframe as dd
from joblib import Memory

warnings.filterwarnings('ignore')

# Configuração de cache para dados
cachedir = Path('cache')
memory = Memory(cachedir, verbose=0)

# Schema para validação de config.yaml
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "paths": {"type": "object"},
        "preprocessing": {"type": "object"},
        "models": {"type": "object"},
        "training": {"type": "object"},
        "mlflow": {"type": "object"},
        "random_state": {"type": "integer"},
    },
    "required": ["paths", "preprocessing", "models", "training", "mlflow"]
}


# =====================================================================================
# 🏗️ CAMADA DE DOMÍNIO (DDD) - LÓGICA DE NEGÓCIO CENTRAL
# =====================================================================================

class ModelType(Enum):
    """Define os tipos de modelos suportados, representando o domínio do problema."""
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "lof"
    ONE_CLASS_SVM = "one_class_svm"  # Novo modelo adicionado


@dataclass
class ModelMetrics:
    """Entidade que representa as métricas de avaliação de um modelo."""
    model_type: ModelType
    training_time: float
    inference_time: float
    memory_usage_mb: float
    anomaly_rate: float
    feature_count: int
    sample_count: int
    cpu_usage_percent: float = 0.0  # Métrica nova: uso de CPU
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Converte a entidade de métricas para um dicionário para logging."""
        return {
            'model_type': self.model_type.value,
            'training_time_seconds': self.training_time,
            'inference_time_ms': self.inference_time,
            'memory_usage_mb': self.memory_usage_mb,
            'anomaly_rate': self.anomaly_rate,
            'feature_count': self.feature_count,
            'sample_count': self.sample_count,
            'cpu_usage_percent': self.cpu_usage_percent,
            'timestamp': self.timestamp.isoformat()
        }


# =====================================================================================
# 🔧 CAMADA DE APLICAÇÃO - CASOS DE USO E ORQUESTRAÇÃO
# =====================================================================================

# --- Observer Pattern ---

class TrainingEvent(Enum):
    """Eventos no ciclo de vida do treinamento."""
    PIPELINE_START = "pipeline_start"
    DATA_LOADING_START = "data_loading_start"
    DATA_LOADING_COMPLETE = "data_loading_complete"
    TRAINING_START = "training_start"
    TRAINING_COMPLETE = "training_complete"
    MODEL_VALIDATED = "model_validated"
    MODEL_SAVED = "model_saved"
    MLFLOW_LOGGING_COMPLETE = "mlflow_logging_complete"
    PIPELINE_COMPLETE = "pipeline_complete"
    PIPELINE_FAILED = "pipeline_failed"


@runtime_checkable
class TrainingObserver(Protocol):
    """Interface para os Observadores do pipeline de treino."""

    def update(self, event: TrainingEvent, data: Dict[str, Any]):
        ...


class Subject:
    """O Sujeito (ou Observável) que notifica os observadores sobre eventos."""

    def __init__(self):
        self._observers: List[TrainingObserver] = []

    def attach(self, observer: TrainingObserver):
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: TrainingObserver):
        self._observers.remove(observer)

    def notify(self, event: TrainingEvent, data: Dict[str, Any]):
        for observer in self._observers:
            observer.update(event, data)


# --- Strategy Pattern ---

@runtime_checkable
class TrainingStrategy(Protocol):
    """Interface para as Estratégias de Treinamento de Modelo."""

    def train(self, X: Union[pd.DataFrame, dd.DataFrame]) -> Any:
        ...

    def validate(self, model: Any, X: Union[pd.DataFrame, dd.DataFrame]) -> ModelMetrics:
        ...

    def tune_hyperparams(self, X: Union[pd.DataFrame, dd.DataFrame], trial: optuna.Trial) -> Dict[str, Any]:
        """Método opcional para tuning de hiperparâmetros."""
        ...


# --- Repository Pattern (DDD) ---

@runtime_checkable
class DataRepository(Protocol):
    """Interface para o Repositório de Dados."""

    def get_prepared_data(self) -> Tuple[Union[pd.DataFrame, dd.DataFrame], Union[pd.DataFrame, dd.DataFrame]]:
        ...


# =====================================================================================
# 🏭 CAMADA DE INFRAESTRUTURA - IMPLEMENTAÇÕES CONCRETAS
# =====================================================================================

class AdvancedLogger:
    """Implementação do logger, agora mais simples pois a formatação é um detalhe de infra."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - [TrustShield] - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log(self, level: int, message: str):
        self.logger.log(level, message)


class ConsoleLogObserver(TrainingObserver):
    """Observador que loga eventos importantes para o console."""

    def __init__(self, logger: AdvancedLogger):
        self.logger = logger

    def update(self, event: TrainingEvent, data: Dict[str, Any]):
        messages = {
            TrainingEvent.PIPELINE_START: f"🚀 === INICIANDO PIPELINE DE TREINO (ID: {data['experiment_id'][:8]}) ===",
            TrainingEvent.DATA_LOADING_START: "📁 Carregando e preparando dados...",
            TrainingEvent.DATA_LOADING_COMPLETE: f"✅ Dados prontos: {data['train_samples']:,} para treino, {data['test_samples']:,} para teste.",
            TrainingEvent.TRAINING_START: f"\n{'=' * 60}\n🎯 TREINANDO MODELO: {data['model_type'].value.upper()}\n{'=' * 60}",
            TrainingEvent.TRAINING_COMPLETE: f"✅ Modelo {data['model_type'].value} treinado com sucesso.",
            TrainingEvent.MODEL_VALIDATED: f"📊 Métricas: Anomalias={data['metrics'].anomaly_rate:.4f} | Inferência={data['metrics'].inference_time:.1f}ms | CPU={data['metrics'].cpu_usage_percent:.1f}%",
            TrainingEvent.MODEL_SAVED: f"💾 Artefato do modelo salvo em: {data['model_path']}",
            TrainingEvent.PIPELINE_COMPLETE: f"\n{'=' * 60}\n🎉 PIPELINE CONCLUÍDO COM SUCESSO em {data['total_time']:.2f}s\n{'=' * 60}",
            TrainingEvent.PIPELINE_FAILED: f"❌ ERRO CRÍTICO NO PIPELINE: {data['error']}",
        }
        if message := messages.get(event):
            self.logger.log(logging.INFO, message)


class MLflowObserver(TrainingObserver):
    """Observador que lida com toda a comunicação com o MLflow."""

    def __init__(self, experiment_name: str, config_path: str):
        self.experiment_name = experiment_name
        self.run_id = None
        self.config_path = config_path

    def update(self, event: TrainingEvent, data: Dict[str, Any]):
        if event == TrainingEvent.TRAINING_START:
            run_name = f"{data['model_type'].value}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            mlflow.start_run(run_name=run_name)
            self.run_id = mlflow.active_run().info.run_id
            mlflow.log_params(data['params'])
            mlflow.log_params({"train_samples": data['train_samples'], "feature_count": data['feature_count']})
            mlflow.log_artifact(self.config_path)  # Loga o config.yaml como artefato

        elif event == TrainingEvent.MODEL_VALIDATED:
            metrics_dict = data['metrics'].to_dict()
            numeric_metrics = {k: v for k, v in metrics_dict.items() if isinstance(v, (int, float))}
            mlflow.log_metrics(numeric_metrics)
            mlflow.set_tag("model_type", metrics_dict['model_type'])

        elif event == TrainingEvent.MLFLOW_LOGGING_COMPLETE:
            model = data['model']
            model_type = data['model_type']
            model_path = data['model_path']
            X_train = data['X_train']

            # Loga schema de features
            schema_path = data.get('schema_path')
            if schema_path:
                mlflow.log_artifact(schema_path)

            if isinstance(model, BaseEstimator) and hasattr(model, 'predict'):
                input_example = X_train.head(5) if isinstance(X_train, dd.DataFrame) else X_train.sample(n=5,
                                                                                                         random_state=42)
                input_example = input_example.compute() if isinstance(input_example, dd.DataFrame) else input_example
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=f"TrustShield-{model_type.value}",
                    input_example=input_example
                )
            else:
                mlflow.log_artifact(model_path, artifact_path="model_artifact")
                mlflow.register_model(
                    model_uri=f"runs:/{self.run_id}/model_artifact/{model_path.name}",
                    name=f"TrustShield-{model_type.value}"
                )
            mlflow.set_tag("status", "success")
            mlflow.end_run()

        elif event == TrainingEvent.PIPELINE_FAILED:
            if mlflow.active_run():
                mlflow.set_tag("status", "failed")
                mlflow.end_run(status=mlflow.entities.RunStatus.FAILED)  # Rollback aprimorado


class IsolationForestStrategy(TrainingStrategy):
    """Estratégia de treino para o Isolation Forest."""

    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def train(self, X: Union[pd.DataFrame, dd.DataFrame]) -> BaseEstimator:
        if isinstance(X, dd.DataFrame):
            X = X.compute()
        model = IsolationForest(**self.params)
        model.fit(X.astype('float32'))
        return model

    def validate(self, model: BaseEstimator, X: Union[pd.DataFrame, dd.DataFrame]) -> ModelMetrics:
        if isinstance(X, dd.DataFrame):
            X = X.compute()
        start_time = time.time()
        predictions = model.predict(X)
        inference_time = (time.time() - start_time) * 1000
        cpu_usage = psutil.cpu_percent(interval=0.1)
        return ModelMetrics(
            model_type=ModelType.ISOLATION_FOREST,
            training_time=0,  # O tempo de treino é medido pelo orquestrador
            inference_time=inference_time,
            memory_usage_mb=psutil.Process().memory_info().rss / (1024 ** 2),
            anomaly_rate=np.sum(predictions == -1) / len(predictions),
            feature_count=len(X.columns),
            sample_count=len(X),
            cpu_usage_percent=cpu_usage
        )

    def tune_hyperparams(self, X: Union[pd.DataFrame, dd.DataFrame], trial: optuna.Trial) -> Dict[str, Any]:
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_samples': trial.suggest_float('max_samples', 0.5, 1.0),
            'contamination': trial.suggest_float('contamination', 0.01, 0.1)
        }


class LOFStrategy(TrainingStrategy):
    """Estratégia de treino para o Local Outlier Factor."""

    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def train(self, X: Union[pd.DataFrame, dd.DataFrame]) -> BaseEstimator:
        if isinstance(X, dd.DataFrame):
            X = X.compute()
        return LocalOutlierFactor(**self.params)

    def validate(self, model: BaseEstimator, X: Union[pd.DataFrame, dd.DataFrame]) -> ModelMetrics:
        if isinstance(X, dd.DataFrame):
            X = X.compute()
        start_time = time.time()
        predictions = model.fit_predict(X)
        inference_time = (time.time() - start_time) * 1000
        cpu_usage = psutil.cpu_percent(interval=0.1)
        return ModelMetrics(
            model_type=ModelType.LOCAL_OUTLIER_FACTOR,
            training_time=inference_time,  # Para LOF, treino e inferência são um só
            inference_time=inference_time,
            memory_usage_mb=psutil.Process().memory_info().rss / (1024 ** 2),
            anomaly_rate=np.sum(predictions == -1) / len(predictions),
            feature_count=len(X.columns),
            sample_count=len(X),
            cpu_usage_percent=cpu_usage
        )

    def tune_hyperparams(self, X: Union[pd.DataFrame, dd.DataFrame], trial: optuna.Trial) -> Dict[str, Any]:
        return {
            'n_neighbors': trial.suggest_int('n_neighbors', 10, 50),
            'contamination': trial.suggest_float('contamination', 0.01, 0.1)
        }


class OneClassSVMStrategy(TrainingStrategy):
    """Estratégia de treino para o One-Class SVM (novo modelo)."""

    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def train(self, X: Union[pd.DataFrame, dd.DataFrame]) -> BaseEstimator:
        if isinstance(X, dd.DataFrame):
            X = X.compute()
        model = OneClassSVM(**self.params)
        model.fit(X.astype('float32'))
        return model

    def validate(self, model: BaseEstimator, X: Union[pd.DataFrame, dd.DataFrame]) -> ModelMetrics:
        if isinstance(X, dd.DataFrame):
            X = X.compute()
        start_time = time.time()
        predictions = model.predict(X)
        inference_time = (time.time() - start_time) * 1000
        cpu_usage = psutil.cpu_percent(interval=0.1)
        return ModelMetrics(
            model_type=ModelType.ONE_CLASS_SVM,
            training_time=0,
            inference_time=inference_time,
            memory_usage_mb=psutil.Process().memory_info().rss / (1024 ** 2),
            anomaly_rate=np.sum(predictions == -1) / len(predictions),
            feature_count=len(X.columns),
            sample_count=len(X),
            cpu_usage_percent=cpu_usage
        )

    def tune_hyperparams(self, X: Union[pd.DataFrame, dd.DataFrame], trial: optuna.Trial) -> Dict[str, Any]:
        return {
            'nu': trial.suggest_float('nu', 0.01, 0.1),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'sigmoid']),
            'gamma': trial.suggest_float('gamma', 0.01, 0.1)
        }


class ModelTrainerFactory:
    """Factory que cria a estratégia de treino apropriada."""

    @staticmethod
    def create_strategy(model_type: ModelType, config: Dict[str, Any]) -> TrainingStrategy:
        params = config.get('models', {}).get(model_type.value, {}).get('params', {})
        params.update({'n_jobs': -1, 'random_state': config.get('random_state', 42)})

        strategies = {
            ModelType.ISOLATION_FOREST: IsolationForestStrategy,
            ModelType.LOCAL_OUTLIER_FACTOR: LOFStrategy,
            ModelType.ONE_CLASS_SVM: OneClassSVMStrategy,
        }
        strategy_class = strategies.get(model_type)
        if not strategy_class:
            raise ValueError(f"Estratégia não encontrada para {model_type}")
        return strategy_class(params)


class ParquetDataRepository(DataRepository):
    """Repositório que busca e prepara os dados de um arquivo Parquet, com suporte a Dask e cache."""

    def __init__(self, config: Dict[str, Any], project_root: Path, use_dask: bool = False):
        self.config = config
        self.project_root = project_root
        self.use_dask = use_dask

    @memory.cache
    def get_prepared_data(self) -> Tuple[Union[pd.DataFrame, dd.DataFrame], Union[pd.DataFrame, dd.DataFrame]]:
        data_path = self.project_root / self.config['paths']['data']['featured_dataset']

        # Carregamento com retry
        for attempt in range(3):
            try:
                if self.use_dask:
                    df = dd.read_parquet(data_path, engine='pyarrow')
                else:
                    df = pd.read_parquet(data_path, engine='pyarrow')
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(1)  # Retry delay

        # Amostragem para controle de memória
        frac = self.config.get('preprocessing', {}).get('sample_frac', 0.01)
        if self.use_dask:
            df = df.sample(frac=frac, random_state=self.config.get('random_state', 42))
        else:
            df = df.sample(frac=frac, random_state=self.config.get('random_state', 42))

        features_to_drop = self.config.get('preprocessing', {}).get('features_to_drop', [])
        X = df.drop(columns=features_to_drop, errors='ignore')

        categorical_features = self.config.get('preprocessing', {}).get('categorical_features', [])
        existing_categorical = [col for col in categorical_features if col in X.columns]
        if existing_categorical:
            if self.use_dask:
                X = dd.get_dummies(X, columns=existing_categorical, drop_first=True, dtype='int8')
            else:
                X = pd.get_dummies(X, columns=existing_categorical, drop_first=True, dtype='int8')

        X = X.fillna(0).astype('float32')

        schema_path = self._save_feature_schema(X.columns.tolist())

        test_size = self.config.get('training', {}).get('test_size', 0.15)
        if self.use_dask:
            X_train, X_test = X.random_split([1 - test_size, test_size],
                                             random_state=self.config.get('random_state', 42))
        else:
            X_train, X_test = train_test_split(
                X,
                test_size=test_size,
                random_state=self.config.get('random_state', 42)
            )
        return X_train, X_test

    def _save_feature_schema(self, feature_names: List[str]) -> Path:
        schema_path = self.project_root / 'outputs' / 'feature_schema.json'
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        schema = {
            'feature_names': feature_names,
            'version': self.config.get('preprocessing', {}).get('feature_store_version', 'v1.0')
        }
        with open(schema_path, 'w') as f:
            json.dump(schema, f, indent=2)
        return schema_path


# =====================================================================================
# 🎼 ORQUESTRADOR - O SERVIÇO PRINCIPAL DA APLICAÇÃO
# =====================================================================================

class AdvancedTrustShieldTrainer(Subject):
    """Orquestrador principal do pipeline de treinamento."""

    def __init__(self, config_path: str, use_dask: bool = False, tune: bool = False, n_trials: int = 10):
        super().__init__()
        self.project_root = Path(__file__).resolve().parents[2]
        self.config = self._load_and_validate_config(config_path)
        self.experiment_id = str(uuid.uuid4())
        self.use_dask = use_dask
        self.tune = tune
        self.n_trials = n_trials

        # Configuração da infraestrutura e observadores
        self.logger = AdvancedLogger('TrustShield')
        self.data_repository = ParquetDataRepository(self.config, self.project_root, use_dask=self.use_dask)
        self.attach(ConsoleLogObserver(self.logger))
        self.attach(MLflowObserver(self.config.get('mlflow', {}).get('experiment_name', 'TrustShield'), config_path))

        self._setup_environment()
        self._setup_signal_handlers()

    def _load_and_validate_config(self, config_path: str) -> Dict[str, Any]:
        with open(self.project_root / config_path, 'r') as f:
            config = yaml.safe_load(f)
        try:
            validate(instance=config, schema=CONFIG_SCHEMA)
        except ValidationError as e:
            raise ValueError(f"Configuração inválida: {e}")
        return config

    def _setup_environment(self):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment(self.config.get('mlflow', {}).get('experiment_name', 'TrustShield'))
        os.environ['OMP_NUM_THREADS'] = '4'
        os.environ['MKL_NUM_THREADS'] = '4'

    def _setup_signal_handlers(self):
        def handler(signum, frame):
            self.logger.log(logging.INFO, f"🛑 Sinal {signum} recebido. Finalizando...")
            if mlflow.active_run():
                mlflow.end_run()
            sys.exit(0)

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def run_pipeline(self, model_types_str: List[str]):
        start_time = time.time()
        try:
            self.notify(TrainingEvent.PIPELINE_START, {"experiment_id": self.experiment_id})

            self.notify(TrainingEvent.DATA_LOADING_START, {})
            X_train, X_test = self.data_repository.get_prepared_data()
            train_samples = len(X_train) if isinstance(X_train, pd.DataFrame) else X_train.shape[0].compute()
            test_samples = len(X_test) if isinstance(X_test, pd.DataFrame) else X_test.shape[0].compute()
            self.notify(TrainingEvent.DATA_LOADING_COMPLETE,
                        {"train_samples": train_samples, "test_samples": test_samples})

            model_types = [ModelType(m) for m in model_types_str if m in ModelType._value2member_map_]

            for model_type in model_types:
                self._train_and_log_model(model_type, X_train, X_test)

            total_time = time.time() - start_time
            self.notify(TrainingEvent.PIPELINE_COMPLETE, {"total_time": total_time})

        except Exception as e:
            self.notify(TrainingEvent.PIPELINE_FAILED, {"error": str(e)})
            self.logger.log(logging.ERROR, f"Erro fatal no pipeline: {e}")
            raise
        finally:
            gc.collect()

    def _train_and_log_model(self, model_type: ModelType, X_train: Union[pd.DataFrame, dd.DataFrame],
                             X_test: Union[pd.DataFrame, dd.DataFrame]):
        params = self.config.get('models', {}).get(model_type.value, {}).get('params', {})

        if self.tune:
            params = self._tune_hyperparams(model_type, X_train, params)

        self.notify(TrainingEvent.TRAINING_START, {
            "model_type": model_type,
            "params": params,
            "train_samples": len(X_train) if isinstance(X_train, pd.DataFrame) else X_train.shape[0].compute(),
            "feature_count": len(X_train.columns)
        })

        strategy = ModelTrainerFactory.create_strategy(model_type, self.config)

        train_start = time.time()
        model = strategy.train(X_train)
        training_time = time.time() - train_start

        self.notify(TrainingEvent.TRAINING_COMPLETE, {"model_type": model_type, "training_time": training_time})

        metrics = strategy.validate(model, X_test)
        metrics.training_time = training_time  # Atualiza o tempo de treino na métrica
        self.notify(TrainingEvent.MODEL_VALIDATED, {"metrics": metrics, "model_type": model_type})

        model_path = self._save_artifact(model, model_type)
        schema_path = self.project_root / 'outputs' / 'feature_schema.json'
        self.notify(TrainingEvent.MODEL_SAVED, {"model_path": model_path})

        self.notify(TrainingEvent.MLFLOW_LOGGING_COMPLETE, {
            "model": model,
            "model_type": model_type,
            "model_path": model_path,
            "X_train": X_train,
            "schema_path": schema_path
        })
        gc.collect()

    def _tune_hyperparams(self, model_type: ModelType, X_train: Union[pd.DataFrame, dd.DataFrame],
                          default_params: Dict[str, Any]) -> Dict[str, Any]:
        strategy = ModelTrainerFactory.create_strategy(model_type, self.config)

        def objective(trial):
            params = strategy.tune_hyperparams(X_train, trial)
            model = strategy.train(X_train)
            metrics = strategy.validate(model, X_train)  # Usa X_train para tuning rápido
            return metrics.anomaly_rate  # Objetivo: minimizar taxa de anomalias (ajuste conforme necessário)

        study = optuna.create_study(direction="minimize")
        mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri())
        study.optimize(objective, n_trials=self.n_trials, callbacks=[mlflow_callback])
        return {**default_params, **study.best_params}

    def _save_artifact(self, model: Any, model_type: ModelType) -> Path:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"{model_type.value}_{timestamp}.joblib"
        model_path = self.project_root / 'outputs' / 'models' / model_name
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path, compress=3)
        return model_path


# =====================================================================================
# 🚀 PONTO DE ENTRADA DA APLICAÇÃO
# =====================================================================================

def main():
    parser = argparse.ArgumentParser(description="Sistema de Treinamento TrustShield Enterprise")
    parser.add_argument(
        "--model",
        type=str,
        default="isolation_forest",
        help="Modelo(s) para treinar (ex: isolation_forest,lof,one_class_svm)."
    )
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--dask", action="store_true", help="Usar Dask para datasets grandes.")
    parser.add_argument("--tune", action="store_true", help="Ativar hyperparameter tuning com Optuna.")
    parser.add_argument("--n-trials", type=int, default=10, help="Número de trials para tuning.")
    args = parser.parse_args()

    try:
        model_types_to_train = [m.strip() for m in args.model.split(",")]
        trainer = AdvancedTrustShieldTrainer(args.config, use_dask=args.dask, tune=args.tune, n_trials=args.n_trials)
        trainer.run_pipeline(model_types_to_train)
        sys.exit(0)
    except Exception as e:
        print(f"❌ ERRO CRÍTICO: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
