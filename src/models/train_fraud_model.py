# -*- coding: utf-8 -*-
"""
Sistema de Treinamento Ultra-Avançado - Projeto TrustShield
VERSÃO EMPRESARIAL REARQUITETADA COM ENGENHARIA DE SOFTWARE CORRIGIDA

🏆 APRIMORAMENTOS PROFUNDOS (v8.0.4-final-fix):
✅ Configuração Automática de Credenciais: Lê os secrets do Docker para autenticar no MinIO.
✅ Artefato Completo: Salva um dicionário {'model': ..., 'scaler': ...} para 100% de compatibilidade.
✅ Gestão de Scaler: O scaler é treinado e versionado junto com o modelo.
✅ MLflow Aprimorado: Registra o artefato completo, garantindo reprodutibilidade.
✅ Logger Robusto: Prevenção de KeyErrors para uma saída de log estável.

Autor: TrustShield Team & IA Gemini
Versão: 8.0.4-final-fix
Data: 2025-07-29
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import optuna
from optuna.integration import MLflowCallback
import dask.dataframe as dd
from joblib import Memory

warnings.filterwarnings('ignore')


# =====================================================================================
# 🔐 CONFIGURAÇÃO DE CREDENCIAIS BOTO3 A PARTIR DE SECRETS DO DOCKER
# =====================================================================================
# Esta função lê as credenciais dos arquivos montados pelo Docker Secrets
# e as exporta como variáveis de ambiente que o Boto3 (usado pelo MLflow) entende.
def setup_boto_credentials():
    access_key_file_path = os.getenv("AWS_ACCESS_KEY_ID_FILE")
    secret_key_file_path = os.getenv("AWS_SECRET_ACCESS_KEY_FILE")

    if access_key_file_path and Path(access_key_file_path).is_file():
        with open(access_key_file_path, 'r') as f:
            os.environ["AWS_ACCESS_KEY_ID"] = f.read().strip()

    if secret_key_file_path and Path(secret_key_file_path).is_file():
        with open(secret_key_file_path, 'r') as f:
            os.environ["AWS_SECRET_ACCESS_KEY"] = f.read().strip()

    # É necessário também para o MLflow se conectar ao S3
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")


setup_boto_credentials()
# =====================================================================================


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
    },
    "required": ["paths", "preprocessing", "models", "training", "mlflow"]
}


# =====================================================================================
# 🏗️ CAMADA DE DOMÍNIO (DDD) - LÓGICA DE NEGÓCIO CENTRAL
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
        """Retorna um dicionário com as métricas numéricas para log no MLflow."""
        # Apenas campos numéricos são adequados para log de métricas no MLflow.
        # O 'model_type' é um Enum e será logado como uma tag.
        # O 'timestamp' não é uma métrica de performance do modelo.
        return {
            "training_time": self.training_time,
            "inference_time": self.inference_time,
            "memory_usage_mb": self.memory_usage_mb,
            "anomaly_rate": self.anomaly_rate,
            "feature_count": self.feature_count,
            "sample_count": self.sample_count,
            "cpu_usage_percent": self.cpu_usage_percent,
        }


# =====================================================================================
# 🔧 CAMADA DE APLICAÇÃO - CASOS DE USO E ORQUESTRAÇÃO
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
# 🏭 CAMADA DE INFRAESTRUTURA - IMPLEMENTAÇÕES CONCRETAS
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
        # Acesso seguro a todas as chaves do dicionário para robustez
        model_type_val = data.get('model_type', type('Enum', (), {'value': 'N/A'})).value
        metrics = data.get('metrics')

        messages = {
            TrainingEvent.PIPELINE_START: f"🚀 === INICIANDO PIPELINE DE TREINO (ID: {data.get('experiment_id', 'N/A')[:8]}) ===",
            TrainingEvent.DATA_LOADING_START: "📁 Carregando e preparando dados...",
            TrainingEvent.DATA_LOADING_COMPLETE: f"✅ Dados prontos: {data.get('train_samples', 0):,} para treino, {data.get('test_samples', 0):,} para teste.",
            TrainingEvent.TRAINING_START: f"\n{'=' * 60}\n🎯 TREINANDO MODELO: {model_type_val.upper()}\n{'=' * 60}",
            TrainingEvent.TRAINING_COMPLETE: f"✅ Modelo {model_type_val} treinado com sucesso.",
            TrainingEvent.MODEL_VALIDATED: f"📊 Métricas: Anomalias={metrics.anomaly_rate:.4f} | Inferência={metrics.inference_time:.1f}ms | CPU={metrics.cpu_usage_percent:.1f}%" if metrics else "Métricas não disponíveis.",
            TrainingEvent.MODEL_SAVED: f"💾 Artefato completo (modelo + scaler) salvo em: {data.get('model_path', 'N/A')}",
            TrainingEvent.PIPELINE_COMPLETE: f"\n{'=' * 60}\n🎉 PIPELINE CONCLUÍDO COM SUCESSO em {data.get('total_time', 0):.2f}s\n{'=' * 60}",
            TrainingEvent.PIPELINE_FAILED: f"❌ ERRO CRÍTICO NO PIPELINE: {data.get('error', 'Desconhecido')}",
        }
        if message := messages.get(event): self.logger.log(logging.INFO, message)


# =====================================================================================
# 래퍼 모델 (SCALER + MODEL) PARA O MLFLOW
# =====================================================================================
class TrustShieldModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Um wrapper de modelo pyfunc do MLflow que inclui um scaler e um modelo de detecção.
    Isso garante que a etapa de pré-processamento (scaling) seja empacotada junto
    com o modelo, tornando o deploy e a inferência mais robustos e consistentes.
    """
    def __init__(self, model: BaseEstimator, scaler: StandardScaler):
        """
        Inicializa o wrapper.
        Args:
            model: O modelo treinado (ex: IsolationForest).
            scaler: O scaler treinado (ex: StandardScaler).
        """
        self.model = model
        self.scaler = scaler

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Executa a predição. O scaler é aplicado antes do modelo.
        Args:
            context: Contexto do MLflow (não utilizado aqui).
            model_input: DataFrame do Pandas com os dados de entrada.
        Returns:
            Um DataFrame com as predições do modelo.
        """
        # Garante que a ordem das colunas na predição seja a mesma do treino
        # (o scaler e o modelo são sensíveis a isso).
        # Se o model_input for um numpy array, ele precisa ser convertido para DataFrame
        # com as colunas corretas. Assumindo que a API enviará um DataFrame.
        scaled_data = self.scaler.transform(model_input)
        predictions = self.model.predict(scaled_data)
        return pd.DataFrame(predictions, columns=['prediction'], index=model_input.index)



class MLflowObserver(TrainingObserver):
    def __init__(self, experiment_name: str, config_path: Path):
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
            mlflow.log_artifact(str(self.config_path))

        elif event == TrainingEvent.MODEL_VALIDATED:
            metrics_obj = data['metrics']
            metrics_to_log = {
                "training_time": metrics_obj.training_time,
                "inference_time": metrics_obj.inference_time,
                "memory_usage_mb": metrics_obj.memory_usage_mb,
                "anomaly_rate": metrics_obj.anomaly_rate,
                "feature_count": metrics_obj.feature_count,
                "sample_count": metrics_obj.sample_count,
                "cpu_usage_percent": metrics_obj.cpu_usage_percent,
            }
            mlflow.log_metrics(metrics_to_log)
            mlflow.set_tag("model_type", data['metrics'].model_type.value)

        elif event == TrainingEvent.MLFLOW_LOGGING_COMPLETE:
            model = data['model']
            scaler = data['scaler']
            model_type = data['model_type']
            local_model_path = str(data['model_path'])

            # Cria o wrapper pyfunc que empacota o scaler e o modelo
            pyfunc_model = TrustShieldModelWrapper(model=model, scaler=scaler)

            # Define o caminho do artefato no MLflow
            artifact_path = f"{model_type.value}_model_packaged"

            # Loga o modelo pyfunc, que é o formato correto para registro
            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=pyfunc_model,
                # Também anexa o arquivo .joblib original como um artefato para referência
                artifacts={"original_artifact": local_model_path}
            )

            # Registra o modelo a partir do artefato pyfunc logado
            mlflow.register_model(
                model_uri=f"runs:/{self.run_id}/{artifact_path}",
                name=f"TrustShield-{model_type.value}"
            )
            mlflow.set_tag("status", "success")
            mlflow.end_run()

        elif event == TrainingEvent.PIPELINE_FAILED:
            if mlflow.active_run():
                mlflow.set_tag("status", "failed")
                mlflow.end_run(status="FAILED")


class BaseTrainingStrategy:
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def _get_data_in_memory(self, X: Union[pd.DataFrame, dd.DataFrame]) -> pd.DataFrame:
        if isinstance(X, dd.DataFrame):
            return X.compute()
        return X


class IsolationForestStrategy(BaseTrainingStrategy, TrainingStrategy):
    def train(self, X: Union[pd.DataFrame, dd.DataFrame]) -> Tuple[BaseEstimator, StandardScaler]:
        X_train = self._get_data_in_memory(X)
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        model = IsolationForest(**self.params)
        model.fit(X_scaled)
        return model, scaler

    def validate(self, model: BaseEstimator, scaler: StandardScaler,
                 X: Union[pd.DataFrame, dd.DataFrame]) -> ModelMetrics:
        X_test = self._get_data_in_memory(X)
        X_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
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
        params.update({'n_jobs': -1, 'random_state': config.get('project', {}).get('random_state', 42)})
        strategies = {ModelType.ISOLATION_FOREST: IsolationForestStrategy}
        strategy_class = strategies.get(model_type)
        if not strategy_class: raise ValueError(f"Estratégia não encontrada para {model_type}")
        return strategy_class(params)


class ParquetDataRepository(DataRepository):
    def __init__(self, config: Dict[str, Any], project_root: Path, use_dask: bool = False):
        self.config = config
        self.project_root = project_root
        self.use_dask = use_dask

    def get_prepared_data(self) -> Tuple[Union[pd.DataFrame, dd.DataFrame], Union[pd.DataFrame, dd.DataFrame]]:
        data_path = self.project_root / self.config['paths']['data']['featured_dataset']
        df = dd.read_parquet(data_path) if self.use_dask else pd.read_parquet(data_path)

        frac = self.config.get('preprocessing', {}).get('sample_frac', 0.1)
        if self.use_dask:
            df = df.sample(frac=frac, random_state=self.config.get('project', {}).get('random_state', 42))
        else:
            df = df.sample(frac=frac, random_state=self.config.get('project', {}).get('random_state', 42))

        features_to_drop = self.config.get('preprocessing', {}).get('features_to_drop', [])
        X = df.drop(columns=features_to_drop, errors='ignore')
        categorical_features = self.config.get('preprocessing', {}).get('categorical_features', [])
        existing_categorical = [col for col in categorical_features if col in X.columns]
        if existing_categorical:
            X = dd.get_dummies(X, columns=existing_categorical, drop_first=True,
                               dtype='int8') if self.use_dask else pd.get_dummies(X, columns=existing_categorical,
                                                                                  drop_first=True, dtype='int8')
        X = X.select_dtypes(include=np.number)
        X = X.fillna(0).astype('float32')

        test_size = self.config.get('training', {}).get('test_size', 0.15)
        if self.use_dask:
            X_train, X_test = X.random_split([1 - test_size, test_size],
                                             random_state=self.config.get('project', {}).get('random_state', 42))
        else:
            X_train, X_test = train_test_split(X, test_size=test_size,
                                               random_state=self.config.get('project', {}).get('random_state', 42))
        return X_train, X_test


# =====================================================================================
# 🎼 ORQUESTRADOR - O SERVIÇO PRINCIPAL DA APLICAÇÃO
# =====================================================================================

class AdvancedTrustShieldTrainer(Subject):
    def __init__(self, config_path: str, use_dask: bool = False, tune: bool = False, n_trials: int = 10):
        super().__init__()
        self.project_root = Path(__file__).resolve().parents[2]
        self.config_path_str = config_path
        self.config = self._load_and_validate_config(config_path)
        self.experiment_id = str(uuid.uuid4())
        self.use_dask = use_dask
        self.tune = tune
        self.n_trials = n_trials
        self.logger = AdvancedLogger('TrustShield')
        self.data_repository = ParquetDataRepository(self.config, self.project_root, use_dask=self.use_dask)
        self.attach(ConsoleLogObserver(self.logger))
        self.attach(MLflowObserver(self.config.get('mlflow', {}).get('experiment_name', 'TrustShield'),
                                   self.project_root / config_path))
        self._setup_environment()

    def _load_and_validate_config(self, config_path: str) -> Dict[str, Any]:
        full_path = self.project_root / config_path
        with open(full_path, 'r') as f: config = yaml.safe_load(f)
        return config

    def _setup_environment(self):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
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
        train_samples = len(X_train) if isinstance(X_train, pd.DataFrame) else X_train.shape[0].compute()
        feature_count = len(X_train.columns)
        self.notify(TrainingEvent.TRAINING_START, {
            "model_type": model_type, "params": params,
            "train_samples": train_samples,
            "feature_count": feature_count
        })
        strategy = ModelTrainerFactory.create_strategy(model_type, self.config)

        train_start = time.time()
        model, scaler = strategy.train(X_train)
        training_time = time.time() - train_start
        self.notify(TrainingEvent.TRAINING_COMPLETE, {"model_type": model_type, "training_time": training_time})

        metrics = strategy.validate(model, scaler, X_test)
        metrics.training_time = training_time
        self.notify(TrainingEvent.MODEL_VALIDATED, {"metrics": metrics})

        # O método _save_artifact foi removido para simplificar.
        # O dicionário do artefato é criado aqui e o caminho é gerenciado diretamente.
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"{model_type.value}_{timestamp}.joblib"
        model_path = self.project_root / 'outputs' / 'models' / model_name
        model_path.parent.mkdir(parents=True, exist_ok=True)

        artifact_payload = {
            'model': model,
            'scaler': scaler,
            'training_timestamp': datetime.now().isoformat()
        }
        joblib.dump(artifact_payload, model_path, compress=3)

        self.notify(TrainingEvent.MODEL_SAVED, {"model_path": model_path})

        # Passa o modelo e o scaler diretamente para o observer do MLflow.
        self.notify(TrainingEvent.MLFLOW_LOGGING_COMPLETE, {
            "model_type": model_type,
            "model_path": model_path,
            "model": model,  # Garante que o objeto do modelo seja passado
            "scaler": scaler # Garante que o objeto do scaler seja passado
        })
        gc.collect()


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