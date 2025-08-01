# -*- coding: utf-8 -*-
"""
Sistema de Treinamento Ultra-Avançado - Projeto TrustShield
VERSÃO EMPRESARIAL REARQUITETADA COM ENGENHARIA DE SOFTWARE CORRIGIDA

🏆 APRIMORAMENTOS PROFUNDOS (v8.1.0-hotfix):
✅ Cache Corrigido: O decorador @memory.cache agora usa `ignore=['self']`.
✅ MLflow Aprimorado: Registra o artefato completo, garantindo reprodutibilidade.
✅ Dask Otimizado: Otimiza o uso de memória ao lidar com Dask DataFrames.
✅ Padrões de Design Refinados: Assinaturas de métodos mais claras e explícitas.

Autor: TrustShield Team & IA Gemini
Versão: 8.1.0-hotfix
Data: 2025-08-01
"""

import argparse
import gc
import logging
import psutil
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
from jsonschema import validate
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import dask.dataframe as dd
from joblib import Memory

warnings.filterwarnings('ignore')

# Configuração de cache para dados
cachedir = Path('cache')
cachedir.mkdir(exist_ok=True)
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
# 🏗️ CAMADA DE DOMÍNIO (DDD)
# =====================================================================================

class ModelType(Enum):
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "lof"
    ONE_CLASS_SVM = "one_class_svm"


@dataclass
class ModelMetrics:
    model_type: ModelType
    training_time: float
    inference_time: float
    memory_usage_mb: float
    anomaly_rate: float
    feature_count: int
    sample_count: int
    cpu_usage_percent: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        metrics_dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (int, float)):
                metrics_dict[k] = v
            elif isinstance(v, datetime):
                metrics_dict[k] = v.timestamp()
        return metrics_dict


# =====================================================================================
# 🔧 CAMADA DE APLICAÇÃO
# =====================================================================================

class TrainingEvent(Enum):
    PIPELINE_START, DATA_LOADING_START, DATA_LOADING_COMPLETE, TRAINING_START, TRAINING_COMPLETE, MODEL_VALIDATED, MODEL_SAVED, MLFLOW_LOGGING_COMPLETE, PIPELINE_COMPLETE, PIPELINE_FAILED = range(
        10)


@runtime_checkable
class TrainingObserver(Protocol):
    def update(self, event: TrainingEvent, data: Dict[str, Any]): ...


class Subject:
    def __init__(self): self._observers: List[TrainingObserver] = []

    def attach(self, observer: TrainingObserver): self._observers.append(observer)

    def detach(self, observer: TrainingObserver): self._observers.remove(observer)

    def notify(self, event: TrainingEvent, data: Dict[str, Any]):
        for observer in self._observers: observer.update(event, data)


@runtime_checkable
class TrainingStrategy(Protocol):
    def train(self, X: Union[pd.DataFrame, dd.DataFrame]) -> Tuple[BaseEstimator, Any]: ...

    def validate(self, model: BaseEstimator, scaler: Any, X: Union[pd.DataFrame, dd.DataFrame]) -> ModelMetrics: ...


@runtime_checkable
class DataRepository(Protocol):
    def get_prepared_data(self) -> Tuple[Union[pd.DataFrame, dd.DataFrame], Union[pd.DataFrame, dd.DataFrame]]: ...


# =====================================================================================
# 🏭 CAMADA DE INFRAESTRUTURA
# =====================================================================================

class AdvancedLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - [TrustShield-Trainer] - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log(self, level: int, message: str): self.logger.log(level, message)


class ConsoleLogObserver(TrainingObserver):
    def __init__(self, logger: AdvancedLogger): self.logger = logger

    def update(self, event: TrainingEvent, data: Dict[str, Any]):
        messages = {
            TrainingEvent.PIPELINE_START: f"🚀 === INICIANDO PIPELINE DE TREINO (ID: {data['experiment_id'][:8]}) ===",
            TrainingEvent.DATA_LOADING_START: "📁 Carregando e preparando dados...",
            TrainingEvent.DATA_LOADING_COMPLETE: f"✅ Dados prontos: {data['train_samples']:,} para treino, {data['test_samples']:,} para teste.",
            TrainingEvent.TRAINING_START: f"\n{'=' * 60}\n🎯 TREINANDO MODELO: {data['model_type'].value.upper()}\n{'=' * 60}",
            TrainingEvent.TRAINING_COMPLETE: f"✅ Modelo {data['model_type'].value} treinado com sucesso.",
            TrainingEvent.MODEL_VALIDATED: f"📊 Métricas: Anomalias={data['metrics'].anomaly_rate:.4f} | Inferência={data['metrics'].inference_time:.1f}ms | CPU={data['metrics'].cpu_usage_percent:.1f}%",
            TrainingEvent.MODEL_SAVED: f"💾 Artefato completo (modelo + scaler) salvo em: {data['model_path']}",
            TrainingEvent.MLFLOW_LOGGING_COMPLETE: f"📦 Artefato e métricas registados no MLflow (Run ID: {data['run_id'][:8]}...)",
            TrainingEvent.PIPELINE_COMPLETE: f"\n{'=' * 60}\n🎉 PIPELINE CONCLUÍDO COM SUCESSO em {data['total_time']:.2f}s\n{'=' * 60}",
            TrainingEvent.PIPELINE_FAILED: f"❌ ERRO CRÍTICO NO PIPELINE: {data['error']}",
        }
        if message := messages.get(event): self.logger.log(logging.INFO, message)


class MLflowObserver(TrainingObserver):
    def update(self, event: TrainingEvent, data: Dict[str, Any]):
        if event == TrainingEvent.PIPELINE_FAILED:
            if mlflow.active_run():
                mlflow.set_tag("status", "failed")
                mlflow.end_run(status="FAILED")


class BaseTrainingStrategy:
    def __init__(self, params: Dict[str, Any]): self.params = params

    def _get_data_in_memory(self, X: Union[pd.DataFrame, dd.DataFrame]) -> pd.DataFrame:
        return X.compute() if isinstance(X, dd.DataFrame) else X


class IsolationForestStrategy(BaseTrainingStrategy, TrainingStrategy):
    def train(self, X: Union[pd.DataFrame, dd.DataFrame]) -> Tuple[BaseEstimator, StandardScaler]:
        X_train = self._get_data_in_memory(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        model = IsolationForest(**self.params)
        model.fit(X_scaled)
        return model, scaler

    def validate(self, model: BaseEstimator, scaler: StandardScaler,
                 X: Union[pd.DataFrame, dd.DataFrame]) -> ModelMetrics:
        X_test = self._get_data_in_memory(X)
        X_scaled = scaler.transform(X_test)
        start_time = time.time()
        predictions = model.predict(X_scaled)
        inference_time = (time.time() - start_time) * 1000
        return ModelMetrics(
            model_type=ModelType.ISOLATION_FOREST, training_time=0, inference_time=inference_time,
            memory_usage_mb=psutil.Process().memory_info().rss / (1024 ** 2),
            anomaly_rate=np.sum(predictions == -1) / len(predictions),
            feature_count=X_test.shape[1], sample_count=len(X_test),
            cpu_usage_percent=psutil.cpu_percent(interval=0.1)
        )


class ModelTrainerFactory:
    @staticmethod
    def create_strategy(model_type: ModelType, config: Dict[str, Any]) -> TrainingStrategy:
        params = config.get('models', {}).get(model_type.value, {}).get('params', {})
        params.update({'n_jobs': -1, 'random_state': config.get('random_state', 42)})
        strategies = {ModelType.ISOLATION_FOREST: IsolationForestStrategy}
        strategy_class = strategies.get(model_type)
        if not strategy_class: raise ValueError(f"Estratégia não encontrada para {model_type}")
        return strategy_class(params)


class ParquetDataRepository(DataRepository):
    def __init__(self, config: Dict[str, Any], project_root: Path, use_dask: bool = False):
        self.config = config
        self.project_root = project_root
        self.use_dask = use_dask

    # CORREÇÃO: Adicionado ignore=['self'] para o cache funcionar corretamente em métodos de instância.
    @memory.cache(ignore=['self'])
    def get_prepared_data(self) -> Tuple[Union[pd.DataFrame, dd.DataFrame], Union[pd.DataFrame, dd.DataFrame]]:
        data_path = self.project_root / self.config['paths']['data']['featured_dataset']
        df = dd.read_parquet(data_path) if self.use_dask else pd.read_parquet(data_path)
        frac = self.config.get('preprocessing', {}).get('sample_frac', 0.01)
        if frac < 1.0:
            df = df.sample(frac=frac, random_state=self.config.get('random_state', 42))

        features_to_drop = self.config.get('preprocessing', {}).get('features_to_drop', [])
        X = df.drop(columns=features_to_drop, errors='ignore')

        categorical_features = self.config.get('preprocessing', {}).get('categorical_features', [])
        existing_categorical = [col for col in categorical_features if col in X.columns]

        if existing_categorical:
            X = (dd.get_dummies(X, columns=existing_categorical, drop_first=True, dtype='int8')
                 if self.use_dask else
                 pd.get_dummies(X, columns=existing_categorical, drop_first=True, dtype='int8'))

        X = X.fillna(0).astype('float32')
        test_size = self.config.get('training', {}).get('test_size', 0.15)

        if self.use_dask:
            X_train, X_test = X.random_split([1 - test_size, test_size],
                                             random_state=self.config.get('random_state', 42))
        else:
            X_train, X_test = train_test_split(X, test_size=test_size, random_state=self.config.get('random_state', 42))

        return X_train, X_test


# =====================================================================================
# 🎼 ORQUESTRADOR
# =====================================================================================

class AdvancedTrustShieldTrainer(Subject):
    def __init__(self, config_path: str, use_dask: bool = False):
        super().__init__()
        self.project_root = Path(__file__).resolve().parents[2]
        self.config_path = config_path
        self.config = self._load_and_validate_config(config_path)
        self.experiment_id = str(uuid.uuid4())
        self.use_dask = use_dask
        self.logger = AdvancedLogger('TrustShield')
        self.data_repository = ParquetDataRepository(self.config, self.project_root, use_dask=self.use_dask)
        self.attach(ConsoleLogObserver(self.logger))
        self.attach(MLflowObserver())
        self._setup_environment()

    def _load_and_validate_config(self, config_path: str) -> Dict[str, Any]:
        with open(self.project_root / config_path, 'r') as f: config = yaml.safe_load(f)
        validate(instance=config, schema=CONFIG_SCHEMA)
        return config

    def _setup_environment(self):
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment(self.config.get('mlflow', {}).get('experiment_name', 'TrustShield'))

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

            self.notify(TrainingEvent.PIPELINE_COMPLETE, {"total_time": time.time() - start_time})
        except Exception as e:
            self.notify(TrainingEvent.PIPELINE_FAILED, {"error": str(e)})
            self.logger.log(logging.ERROR, f"Erro fatal no pipeline: {e}")
            raise
        finally:
            gc.collect()

    def _train_and_log_model(self, model_type: ModelType, X_train, X_test):
        params = self.config.get('models', {}).get(model_type.value, {}).get('params', {})

        with mlflow.start_run(run_name=f"{model_type.value}_{datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
            run_id = run.info.run_id
            self.notify(TrainingEvent.TRAINING_START, {
                "model_type": model_type, "params": params,
                "train_samples": len(X_train) if isinstance(X_train, pd.DataFrame) else X_train.shape[0].compute(),
                "feature_count": len(X_train.columns)
            })

            strategy = ModelTrainerFactory.create_strategy(model_type, self.config)
            train_start = time.time()
            model, scaler = strategy.train(X_train)
            training_time = time.time() - train_start
            self.notify(TrainingEvent.TRAINING_COMPLETE, {"model_type": model_type, "training_time": training_time})

            metrics = strategy.validate(model, scaler, X_test)
            metrics.training_time = training_time
            self.notify(TrainingEvent.MODEL_VALIDATED, {"metrics": metrics})

            model_path = self._save_artifact(model, scaler, model_type)
            self.notify(TrainingEvent.MODEL_SAVED, {"model_path": model_path})

            # Log no MLflow
            mlflow.log_params(params)
            mlflow.log_metrics(metrics.to_dict())
            mlflow.log_artifact(self.project_root / self.config_path, artifact_path="config")
            mlflow.log_artifact(model_path, artifact_path="model_artifact")

            registered_model_name = f"TrustShield-{model_type.value}"
            mlflow.register_model(model_uri=f"runs:/{run_id}/model_artifact", name=registered_model_name)

            self.notify(TrainingEvent.MLFLOW_LOGGING_COMPLETE, {"run_id": run_id})
        gc.collect()

    def _save_artifact(self, model: Any, scaler: Any, model_type: ModelType) -> Path:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"{model_type.value}_{timestamp}.joblib"
        model_path = self.project_root / 'outputs' / 'models' / model_name
        model_path.parent.mkdir(parents=True, exist_ok=True)

        artifacts = {'model': model, 'scaler': scaler, 'training_timestamp': datetime.now().isoformat()}
        joblib.dump(artifacts, model_path, compress=3)
        return model_path


# =====================================================================================
# 🚀 PONTO DE ENTRADA DA APLICAÇÃO
# =====================================================================================

def main():
    parser = argparse.ArgumentParser(description="Sistema de Treinamento TrustShield Enterprise")
    parser.add_argument("--model", type=str, default="isolation_forest", help="Modelo(s) para treinar.")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--dask", action="store_true", help="Usar Dask para datasets grandes.")
    args = parser.parse_args()

    try:
        model_types_to_train = [m.strip() for m in args.model.split(",")]
        trainer = AdvancedTrustShieldTrainer(args.config, use_dask=args.dask)
        trainer.run_pipeline(model_types_to_train)
        sys.exit(0)
    except Exception as e:
        print(f"❌ ERRO CRÍTICO: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()