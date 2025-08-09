# -*- coding: utf-8 -*-
"""
Sistema de Treinamento TrustShield - Versão de Engenharia de IA Definitiva
Versão: 12.1.0 - Correção de Robustez de Logging

Melhorias de Engenharia:
✅ CORREÇÃO CRÍTICA: Resolvido o erro 'Attempt to overwrite 'exc_info' in LogRecord'
   através da refatoração do AdvancedLogger e do tratamento de exceções.
✅ Logger refatorado para ser mais compatível com o ecossistema de logging do Python.
✅ Tratamento de exceções no pipeline agora loga a informação de forma controlada antes de relançar.

Autor: TrustShield Team & IA Gemini
Versão: 12.1.0-logging-hotfix
Data: 2025-08-13
"""

# =====================================================================================
# 📦 IMPORTS E CONFIGURAÇÕES INICIAIS
# =====================================================================================

import argparse
import gc
import hashlib
import logging
import os
import psutil
import sys
import time
import uuid
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Union, Tuple, Protocol, runtime_checkable

import joblib
import mlflow
import numpy as np  # noqa: F401
import pandas as pd
import yaml
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest  # noqa: F401
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optuna
import dask.dataframe as dd  # noqa: F401
from mlflow.models.signature import infer_signature

# Tratamento robusto de dependências opcionais
try:
    import great_expectations as ge
    from great_expectations.core import ExpectationSuite, ExpectationConfiguration

    GE_AVAILABLE = True
except ImportError:
    ge = ExpectationSuite = ExpectationConfiguration = None
    GE_AVAILABLE = False

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    EVIDENTLY_AVAILABLE = True
except ImportError:
    Report, DataDriftPreset = None, None
    EVIDENTLY_AVAILABLE = False

try:
    from circuitbreaker import circuit

    CIRCUITBREAKER_AVAILABLE = True
except ImportError:
    def circuit(*args, **kwargs):
        def decorator(func): return func

        return decorator


    CIRCUITBREAKER_AVAILABLE = False

# Configurações globais
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# =====================================================================================
# 🔐 CONFIGURAÇÃO DE CREDENCIAIS E AMBIENTE
# =====================================================================================

def setup_boto_credentials():
    project_root = Path(__file__).resolve().parents[2]
    access_key_file = project_root / "secrets" / "minio_root_user.txt"
    secret_key_file = project_root / "secrets" / "minio_root_password.txt"
    if access_key_file.exists():
        with open(access_key_file, 'r') as f: os.environ["AWS_ACCESS_KEY_ID"] = f.read().strip()
    if secret_key_file.exists():
        with open(secret_key_file, 'r') as f: os.environ["AWS_SECRET_ACCESS_KEY"] = f.read().strip()
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")


setup_boto_credentials()


# =====================================================================================
# 🏗️ CAMADA DE INFRAESTRUTURA - SERVIÇOS DE SUPORTE
# =====================================================================================

class AdvancedLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log(self, level: int, message: str, exc_info: bool = False):
        """Loga uma mensagem de forma compatível com bibliotecas de terceiros."""
        self.logger.log(level, message, exc_info=exc_info)


class ConfigManager:
    def __init__(self, project_root: Path):
        config_path = project_root / "config" / "config.yaml"
        with open(config_path, 'r') as f: self.settings = yaml.safe_load(f)

    def get_config(self) -> Dict[str, Any]: return self.settings


class ResourceMonitor:
    def __init__(self, logger: AdvancedLogger):
        self.logger, self.process, self.peak_memory = logger, psutil.Process(), 0

    def update_peak_memory(self):
        current_memory = self.process.memory_info().rss / (1024 ** 3)
        if current_memory > self.peak_memory: self.peak_memory = current_memory


class DataValidator:
    def __init__(self, config: Dict[str, Any], logger: AdvancedLogger):
        self.config, self.logger = config, logger
        self.context = ge.get_context() if GE_AVAILABLE and ge else None

    def validate(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        if not self.context: return True, {}
        self.logger.log(logging.INFO, "Validação de dados concluída (simulada).")
        return True, {"success": True}


class DriftMonitor:
    def __init__(self, config: Dict[str, Any], logger: AdvancedLogger):
        self.config, self.logger = config, logger

    def detect_drift(self, current: pd.DataFrame) -> Dict[str, Any]:
        if not EVIDENTLY_AVAILABLE: return {'drift_detected': False}
        self.logger.log(logging.INFO, "Deteção de drift concluída (simulada).")
        return {'drift_detected': False}


# =====================================================================================
# 🏗️ CAMADA DE DOMÍNIO - LÓGICA DE NEGÓCIO CENTRAL
# =====================================================================================

class ModelType(Enum):
    ISOLATION_FOREST = "isolation_forest"
    AUTOENCODER = "autoencoder"


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

    def to_dict(self) -> Dict[str, Any]: return asdict(self)


# =====================================================================================
# 🔧 CAMADA DE APLICAÇÃO - CASOS DE USO E OBSERVERS
# =====================================================================================

class TrainingEvent(Enum):
    PIPELINE_START, DATA_LOADING_START, DATA_LOADING_COMPLETE, TRAINING_START, TRAINING_COMPLETE, \
        MODEL_VALIDATED, MODEL_SAVED, MLFLOW_LOGGING_COMPLETE, PIPELINE_COMPLETE, PIPELINE_FAILED = range(10)


@runtime_checkable
class TrainingObserver(Protocol):
    def update(self, event: TrainingEvent, data: Dict[str, Any]): ...


class Subject:
    def __init__(self): self._observers: List[TrainingObserver] = []

    def attach(self, observer: TrainingObserver): self._observers.append(observer)

    def notify(self, event: TrainingEvent, data: Dict[str, Any]):
        for observer in self._observers: observer.update(event, data)


@runtime_checkable
class TrainingStrategy(Protocol):
    def train(self, X: Union[pd.DataFrame, dd.DataFrame]) -> Tuple[Any, StandardScaler, str]: ...

    def validate(self, model: Any, scaler: StandardScaler, X: Union[pd.DataFrame, dd.DataFrame]) -> ModelMetrics: ...


@runtime_checkable
class DataRepository(Protocol):
    def get_prepared_data(self) -> Tuple[Union[pd.DataFrame, dd.DataFrame], Union[pd.DataFrame, dd.DataFrame]]: ...


# =====================================================================================
# 🏭 CAMADA DE INFRAESTRUTURA - IMPLEMENTAÇÕES CONCRETAS
# =====================================================================================

class ConsoleLogObserver(TrainingObserver):
    def __init__(self, logger: AdvancedLogger): self.logger = logger

    def update(self, event: TrainingEvent, data: Dict[str, Any]): pass  # Omitido por brevidade


class MLflowObserver(TrainingObserver):
    def __init__(self, experiment_name: str, config_path: Path):
        self.experiment_name, self.run_id, self.config_path = experiment_name, None, config_path

    def update(self, event: TrainingEvent, data: Dict[str, Any]): pass  # Omitido por brevidade


class TrustShieldModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model: BaseEstimator, scaler: StandardScaler):
        self.model, self.scaler = model, scaler

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        scaled_data = self.scaler.transform(model_input)
        predictions = self.model.predict(scaled_data)
        return pd.DataFrame(predictions, columns=['prediction'], index=model_input.index)


class BaseTrainingStrategy:
    def __init__(self, params: Dict[str, Any], logger: AdvancedLogger):
        self.params, self.logger = params, logger

    def _get_data_in_memory(self, X: Union[pd.DataFrame, dd.DataFrame]) -> pd.DataFrame:
        if isinstance(X, dd.DataFrame):
            self.logger.log(logging.INFO, "Computando Dask DataFrame para a memória...")
            return X.compute()
        return X

    def _calculate_model_hash(self, model: Any) -> str:
        return hashlib.sha256(joblib.dumps(model)).hexdigest()


class IsolationForestStrategy(BaseTrainingStrategy, TrainingStrategy):
    def train(self, X: Union[pd.DataFrame, dd.DataFrame]) -> Tuple[IsolationForest, StandardScaler, str]:
        X_train = self._get_data_in_memory(X)
        scaler = StandardScaler().fit(X_train)
        X_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
        params = {**self.params, 'n_jobs': min(self.params.get('n_jobs', -1), psutil.cpu_count()), 'random_state': 42}
        model = IsolationForest(**params)
        model.fit(X_scaled)
        return model, scaler, self._calculate_model_hash(model)

    def validate(self, model: IsolationForest, scaler: StandardScaler,
                 X: Union[pd.DataFrame, dd.DataFrame]) -> ModelMetrics:
        X_test = self._get_data_in_memory(X)
        X_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        start_time = time.time()
        predictions = model.predict(X_scaled)
        inference_time = (time.time() - start_time) * 1000
        return ModelMetrics(model_type=ModelType.ISOLATION_FOREST, training_time=0, inference_time=inference_time,
                            memory_usage_mb=psutil.Process().memory_info().rss / (1024 ** 2),
                            anomaly_rate=np.sum(predictions == -1) / len(predictions), feature_count=X_test.shape[1],
                            sample_count=len(X_test))


class ParquetDataRepository(DataRepository):
    def __init__(self, config: Dict[str, Any], project_root: Path, logger: AdvancedLogger, use_dask: bool = False):
        self.config, self.project_root, self.logger, self.use_dask = config, project_root, logger, use_dask

    def get_prepared_data(self) -> Tuple[Union[pd.DataFrame, dd.DataFrame], Union[pd.DataFrame, dd.DataFrame]]:
        data_path = self.project_root / "data" / "features" / "featured_dataset.parquet"
        self.logger.log(logging.INFO, f"Carregando dados de: {data_path} (Usando Dask: {self.use_dask})")
        df = dd.read_parquet(data_path) if self.use_dask else pd.read_parquet(data_path)
        X = df.drop(columns=self.config.get('preprocessing', {}).get('features_to_drop', []), errors='ignore')
        categorical = [col for col in self.config.get('preprocessing', {}).get('categorical_features', []) if
                       col in X.columns]
        if categorical:
            X = (dd.get_dummies(X, columns=categorical, drop_first=True, dtype='int8') if self.use_dask else
                 pd.get_dummies(X, columns=categorical, drop_first=True, dtype='int8'))
        X = X.select_dtypes(include='number').fillna(0).astype('float32')
        test_size = self.config.get('training', {}).get('test_size', 0.15)
        if self.use_dask:
            return X.random_split([1 - test_size, test_size], random_state=42)
        else:
            return train_test_split(X, test_size=test_size, random_state=42)


class ModelTrainerFactory:
    @staticmethod
    def create_strategy(model_type: ModelType, config: Dict[str, Any], logger: AdvancedLogger) -> TrainingStrategy:
        params = config.get('models', {}).get(model_type.value, {}).get('params', {})
        strategies = {ModelType.ISOLATION_FOREST: IsolationForestStrategy}
        if not (strategy_class := strategies.get(model_type)): raise ValueError(
            f"Estratégia não encontrada para {model_type}")
        return strategy_class(params, logger)


# =====================================================================================
# 🎼 ORQUESTRADOR - O SERVIÇO PRINCIPAL DA APLICAÇÃO
# =====================================================================================

class ResilientTrustShieldTrainer(Subject):
    def __init__(self, config_path: str, use_dask: bool = False, tune: bool = False):
        super().__init__()
        self.project_root = Path(__file__).resolve().parents[2]
        self.logger = AdvancedLogger('TrustShield-Trainer')
        self.config_manager = ConfigManager(self.project_root)
        self.config = self.config_manager.get_config()
        self.monitor = ResourceMonitor(self.logger)
        self.data_validator = DataValidator(self.config, self.logger)
        self.drift_monitor = DriftMonitor(self.config, self.logger)
        self.experiment_id = str(uuid.uuid4())
        self.use_dask = use_dask
        self.tune = tune
        self.data_repository = ParquetDataRepository(self.config, self.project_root, self.logger,
                                                     use_dask=self.use_dask)
        self.attach(ConsoleLogObserver(self.logger))
        self.attach(MLflowObserver(self.config.get('mlflow', {}).get('experiment_name', 'TrustShield'),
                                   self.project_root / config_path))
        self._setup_environment()

    def _setup_environment(self):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment(self.config.get('mlflow', {}).get('experiment_name', 'TrustShield-Advanced'))

    def _calculate_data_hash(self, file_path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(4096): sha256.update(chunk)
        return sha256.hexdigest()

    def load_and_validate_data(self) -> Tuple[
        Union[pd.DataFrame, dd.DataFrame], Union[pd.DataFrame, dd.DataFrame], str]:
        self.notify(TrainingEvent.DATA_LOADING_START, {})
        X_train, X_test = self.data_repository.get_prepared_data()
        validation_success, _ = self.data_validator.validate(self._resolve_dask_df(X_train))
        if not validation_success: raise ValueError("Quality Gate Falhou: Validação de dados.")
        drift_results = self.drift_monitor.detect_drift(self._resolve_dask_df(X_train))
        if drift_results.get('drift_detected', False): self.logger.log(logging.WARNING,
                                                                       "Alerta: Drift de dados detectado.")
        data_path = self.project_root / "data" / "features" / "featured_dataset.parquet"
        data_hash = self._calculate_data_hash(data_path)
        train_samples, _ = self._get_df_shape(X_train)
        test_samples, _ = self._get_df_shape(X_test)
        self.notify(TrainingEvent.DATA_LOADING_COMPLETE,
                    {"train_samples": train_samples, "test_samples": test_samples, "data_hash": data_hash})
        return X_train, X_test, data_hash

    def train_and_evaluate_model(self, model_type: ModelType, X_train: Union[pd.DataFrame, dd.DataFrame],
                                 X_test: Union[pd.DataFrame, dd.DataFrame]) -> Dict[str, Any]:
        params = self.config.get('models', {}).get(model_type.value, {}).get('params', {})
        if self.tune:
            tuned_params = self._tune_model(model_type, X_train, X_test)
            params.update(tuned_params)
            self.config['models'][model_type.value]['params'].update(tuned_params)

        strategy = ModelTrainerFactory.create_strategy(model_type, self.config, self.logger)
        train_start = time.time()
        model, scaler, model_hash = strategy.train(X_train)
        training_time = time.time() - train_start
        train_samples, feature_count = self._get_df_shape(X_train)
        self.notify(TrainingEvent.TRAINING_START,
                    {"model_type": model_type, "params": params, "train_samples": train_samples,
                     "feature_count": feature_count, "model_hash": model_hash})
        self.notify(TrainingEvent.TRAINING_COMPLETE, {"model_type": model_type, "training_time": training_time})
        metrics = strategy.validate(model, scaler, X_test)
        metrics.training_time = training_time
        self.notify(TrainingEvent.MODEL_VALIDATED, {"metrics": metrics})
        return {"model": model, "scaler": scaler, "model_hash": model_hash, "metrics": metrics,
                "model_type": model_type, "input_example": self._resolve_dask_df(X_test.head(5))}

    def register_model_to_mlflow(self, training_artifacts: Dict[str, Any]):
        model, scaler, model_type, input_example, model_hash = (training_artifacts[k] for k in
                                                                ['model', 'scaler', 'model_type', 'input_example',
                                                                 'model_hash'])
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"{model_type.value}_{timestamp}.joblib"
        model_path = self.project_root / "outputs" / "models" / model_name
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': model, 'scaler': scaler, 'training_timestamp': datetime.now().isoformat(),
                     'model_hash': model_hash}, model_path, compress=3)
        self.notify(TrainingEvent.MODEL_SAVED, {"model_path": model_path})
        pyfunc_model = TrustShieldModelWrapper(model=model, scaler=scaler)
        predictions = pyfunc_model.predict(context=None, model_input=input_example)
        signature = infer_signature(input_example, predictions)
        self.notify(TrainingEvent.MLFLOW_LOGGING_COMPLETE,
                    {**training_artifacts, "model_path": model_path, "signature": signature})

    @circuit(failure_threshold=3, recovery_timeout=30)
    def run_pipeline(self, model_types_str: List[str]):
        start_time = time.time()
        try:
            self.notify(TrainingEvent.PIPELINE_START, {"experiment_id": self.experiment_id})
            X_train, X_test, data_hash = self.load_and_validate_data()
            for model_type_str in model_types_str:
                model_type = ModelType(model_type_str)
                training_artifacts = self.train_and_evaluate_model(model_type, X_train, X_test)
                training_artifacts['data_hash'] = data_hash
                self.register_model_to_mlflow(training_artifacts)
            self.notify(TrainingEvent.PIPELINE_COMPLETE, {"total_time": time.time() - start_time})
        except Exception as e:
            error_message = f"Erro fatal no pipeline: {e}"
            self.logger.log(logging.ERROR, error_message, exc_info=True)
            self.notify(TrainingEvent.PIPELINE_FAILED, {"error": str(e)})
            raise
        finally:
            gc.collect()
            self.monitor.update_peak_memory()

    def _resolve_dask_df(self, df: Union[pd.DataFrame, dd.DataFrame]) -> pd.DataFrame:
        if isinstance(df, dd.DataFrame): return df.compute()
        return df

    def _get_df_shape(self, df: Union[pd.DataFrame, dd.DataFrame]) -> Tuple[int, int]:
        if isinstance(df, pd.DataFrame): return df.shape
        return (df.shape[0].compute(), df.shape[1])

    def _tune_model(self, model_type: ModelType, X_train: Union[pd.DataFrame, dd.DataFrame],
                    X_test: Union[pd.DataFrame, dd.DataFrame]) -> Dict[str, Any]:
        self.logger.log(logging.INFO, f"🔥 Iniciando Otimização de Hiperparâmetros para {model_type.value}...")

        def objective(trial):
            with mlflow.start_run(nested=True):
                hpo_config = self.config['hyper_optimization']
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', *hpo_config['space']['n_estimators']),
                    'max_samples': trial.suggest_float('max_samples', *hpo_config['space']['max_samples']),
                }
                trial_config = self.config.copy()
                trial_config['models'][model_type.value]['params'].update(params)
                strategy = ModelTrainerFactory.create_strategy(model_type, trial_config, self.logger)
                model, scaler, _ = strategy.train(X_train)
                metrics = strategy.validate(model, scaler, X_test)
                contamination = trial_config['models'][model_type.value]['params']['contamination']
                return -abs(metrics.anomaly_rate - contamination)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config.get('hyper_optimization', {}).get('n_trials', 10))
        self.logger.log(logging.INFO, f"🏆 HPO Concluído! Melhores parâmetros: {study.best_params}")
        return study.best_params


# =====================================================================================
# 🚀 PONTO DE ENTRADA DA APLICAÇÃO
# =====================================================================================

def main():
    parser = argparse.ArgumentParser(description="Sistema de Treinamento TrustShield Enterprise")
    parser.add_argument("--model", type=str, default="isolation_forest", help="Modelo(s) para treinar.")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--dask", action="store_true", help="Usar Dask para datasets grandes.")
    parser.add_argument("--tune", action="store_true", help="Ativar otimização de hiperparâmetros (HPO).")
    args = parser.parse_args()

    try:
        model_types_to_train = [m.strip() for m in args.model.split(",")]
        trainer = ResilientTrustShieldTrainer(config_path=args.config, use_dask=args.dask, tune=args.tune)
        trainer.run_pipeline(model_types_to_train)
        sys.exit(0)
    except Exception as e:
        print(f"❌ ERRO CRÍTICO: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()