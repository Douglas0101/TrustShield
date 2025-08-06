# ==============================================================================
# ARQUIVO: src/models/train_fraud_model.py (OTIMIZADO E REATORADO)
# ==============================================================================
# -*- coding: utf-8 -*-
"""
Sistema de Treinamento TrustShield - ARQUITETURA ORQUESTRADA
Versão: 4.0.0 - High Performance & Robustness

Este script atua como o orquestrador principal do pipeline de treinamento.
Ele delega responsabilidades para módulos especializados, garantindo um fluxo
de trabalho limpo, lógico e alinhado com as melhores práticas de MLOps.

Autor: TrustShield Team & Enhanced by AI Engineering Team
Data: 2025-08-07
"""
import argparse
import hashlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dask.dataframe as dd
import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.models.signature import ModelSignature, infer_signature
from scipy.stats import ks_2samp
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

# --- Configuração do Projeto ---
# Adiciona o root do projeto ao sys.path para importações modulares robustas
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Importa os módulos refatorados
from src.models.validation import ConfigValidator, DataValidator
from src.models.optimization import HybridOptimizer
from src.models.interpretation import ModelInterpreter


# --- Configuração do Logger ---
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - [TrustShield-Trainer] - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


LOGGER = get_logger("Trainer")

# --- Definições de Diretórios ---
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DATA_DIR = DATA_DIR / "features"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
INTERPRETATION_DIR = OUTPUTS_DIR / "interpretation"

# Garante que os diretórios de saída existam
for directory in [MODELS_DIR, INTERPRETATION_DIR]:
    directory.mkdir(exist_ok=True, parents=True)


# --- Estruturas de Dados ---
@dataclass
class Artifacts:
    """Uma classe de dados simples para armazenar os artefactos de treinamento."""
    model: Any
    scaler: Any
    metrics: Dict[str, float]
    params: Dict[str, Any]
    signature: ModelSignature
    feature_names: List[str]


# --- Componentes do Pipeline ---
class DataPipeline:
    """Lida com a ingestão, validação, divisão e deteção de drift dos dados."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_validator = DataValidator(config)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _load_and_validate_data(self, path: Path) -> pd.DataFrame:
        """Carrega os dados usando Dask para eficiência e valida com Pandera."""
        try:
            LOGGER.info(f"Carregando dados de: {path}")
            if not path.exists():
                raise FileNotFoundError(f"Arquivo de dados não encontrado: {path}")

            # Usa Dask para a leitura inicial, depois computa para pandas
            df_dask = dd.read_parquet(path, engine='pyarrow')
            df = df_dask.compute()

            # Valida o schema no dataframe completo agora que está em memória
            self.data_validator.validate_schema(df)
            return df
        except Exception as e:
            LOGGER.error("Falha crítica durante a ingestão e validação de dados.", exc_info=True)
            raise

    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Divide os dados em conjuntos de treino, validação e teste."""
        test_size = self.config['preprocessing']['test_size']
        val_size = self.config['preprocessing']['validation_size']

        # Calcula o tamanho relativo da validação para a segunda divisão
        train_val_size = 1.0 - test_size
        val_relative_size = val_size / train_val_size

        train_val, test = train_test_split(df, test_size=test_size, shuffle=False)
        train, val = train_test_split(train_val, test_size=val_relative_size, shuffle=False)

        LOGGER.info(f"Divisão de dados concluída: {len(train)} treino, {len(val)} validação, {len(test)} teste.")
        return train, val, test

    def _check_for_drift(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict[str, float]:
        """Executa um teste KS para data drift entre os conjuntos de treino e teste."""
        LOGGER.info("Executando verificação de data drift...")
        drift_metrics = {}
        for col in df_train.columns:
            if pd.api.types.is_numeric_dtype(df_train[col]):
                stat, p_value = ks_2samp(df_train[col], df_test[col])
                drift_metrics[f"drift_p_value_{col}"] = p_value
                if p_value < 0.05:
                    LOGGER.warning(f"Drift potencial detetado na coluna '{col}' (p-value: {p_value:.4f})")
        return drift_metrics

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Executa o pipeline de dados completo."""
        data_path = FEATURES_DATA_DIR / self.config['data']['featured_filename']
        df = self._load_and_validate_data(data_path)

        train_df, val_df, test_df = self._split_data(df)

        # Log do hash dos dados para reprodutibilidade
        train_hash = hashlib.sha256(pd.util.hash_pandas_object(train_df, index=True).values).hexdigest()
        mlflow.log_param("train_data_hash", train_hash)

        # Verifica o drift e loga as métricas
        drift_metrics = self._check_for_drift(train_df[self.config['data']['model_features']],
                                              test_df[self.config['data']['model_features']])
        mlflow.log_metrics(drift_metrics)

        return train_df, val_df, test_df


class ModelTrainer:
    """Lida com a lógica de treinamento e avaliação do modelo."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.random_state = config['project']['random_state']
        self.model_features = config['data']['model_features']

    def _get_scaler(self) -> Any:
        """Seleciona o scaler com base na configuração."""
        scaler_type = self.config.get('preprocessing', {}).get('scaler', 'standard')
        return RobustScaler() if scaler_type == 'robust' else StandardScaler()

    def run(self, df_train: pd.DataFrame, df_val: pd.DataFrame, params: Dict[str, Any]) -> Artifacts:
        """Executa o treinamento do modelo e a criação de artefactos."""
        LOGGER.info("Iniciando treinamento do modelo...")

        scaler = self._get_scaler()
        X_train_scaled = scaler.fit_transform(df_train[self.model_features])
        X_val_scaled = scaler.transform(df_val[self.model_features])

        model = IsolationForest(**params, random_state=self.random_state, n_jobs=-1)
        model.fit(X_train_scaled)

        scores_val = -model.decision_function(X_val_scaled)

        metrics = {
            "anomaly_rate_val": float(np.mean(scores_val > np.quantile(scores_val, 0.95))),
            "score_variance_val": float(np.var(scores_val))
        }

        input_example = df_train[self.model_features].head(1)
        signature = infer_signature(input_example, pd.DataFrame(scores_val[:1], columns=["anomaly_score"]))

        LOGGER.info("Treinamento do modelo concluído com sucesso.")
        return Artifacts(
            model=model,
            scaler=scaler,
            metrics=metrics,
            params=params,
            signature=signature,
            feature_names=self.model_features
        )


# --- Função Principal de Orquestração ---
def train_model(config_path: str) -> None:
    """Função principal para orquestrar todo o pipeline de treinamento."""
    LOGGER.info("=" * 60)
    LOGGER.info("PIPELINE DE TREINAMENTO TRUSTSHIELD (ORQUESTRADOR OTIMIZADO)")
    LOGGER.info("=" * 60)

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        ConfigValidator.validate(config)
    except (FileNotFoundError, ValueError) as e:
        LOGGER.error(f"Erro de configuração: {e}", exc_info=True)
        sys.exit(1)

    mlflow.set_experiment(config['mlflow']['experiment_name'])
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        LOGGER.info(f"MLflow Run ID: {run_id}")
        mlflow.log_params(config['project'])
        mlflow.log_artifact(config_path)

        # --- PASSO 1: Pipeline de Dados ---
        LOGGER.info("--- INICIANDO PASSO 1: PIPELINE DE DADOS ---")
        data_pipeline = DataPipeline(config)
        try:
            df_train, df_val, df_test = data_pipeline.run()
        except (RetryError, FileNotFoundError, ValueError) as e:
            LOGGER.error("Falha no pipeline de dados. Abortando treinamento.", exc_info=True)
            sys.exit(1)

        # --- PASSO 2: Otimização de Hiperparâmetros ---
        LOGGER.info("--- INICIANDO PASSO 2: OTIMIZAÇÃO DE HIPERPARÂMETROS ---")
        if config.get('hyper_optimization', {}).get('enabled', False):
            optimizer = HybridOptimizer(config)
            best_params = optimizer.optimize(
                df_train[config['data']['model_features']],
                df_val[config['data']['model_features']]
            )
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        else:
            LOGGER.info("Otimização de hiperparâmetros desabilitada. Usando parâmetros do config.")
            best_params = config['model']['default_params']

        # --- PASSO 3: Treinamento do Modelo ---
        LOGGER.info("--- INICIANDO PASSO 3: TREINAMENTO DO MODELO ---")
        trainer = ModelTrainer(config)
        artifacts = trainer.run(df_train, df_val, best_params)
        mlflow.log_metrics(artifacts.metrics)
        mlflow.log_params(artifacts.params)

        # --- PASSO 4: Interpretação do Modelo ---
        LOGGER.info("--- INICIANDO PASSO 4: INTERPRETAÇÃO DO MODELO ---")
        if config.get('interpretability', {}).get('enabled', False):
            interpreter = ModelInterpreter(
                model=artifacts.model,
                data=df_train[artifacts.feature_names],
                config=config
            )
            interpreter.run_and_save(output_path=str(INTERPRETATION_DIR / run_id))
            mlflow.log_artifacts(str(INTERPRETATION_DIR / run_id), artifact_path="interpretation")

        # --- PASSO 5: Log do Modelo e Artefactos ---
        LOGGER.info("--- INICIANDO PASSO 5: REGISTRO DE ARTEFACTOS ---")
        model_info = mlflow.sklearn.log_model(
            sk_model=artifacts.model,
            artifact_path="model",
            signature=artifacts.signature,
            registered_model_name=config['mlflow'].get('registered_model_name')
        )
        LOGGER.info(f"Modelo registrado com sucesso: {model_info.model_uri}")

        # Salva o scaler junto com o modelo
        joblib.dump(artifacts.scaler, "scaler.joblib")
        mlflow.log_artifact("scaler.joblib", "model")

    LOGGER.info("=" * 60)
    LOGGER.info("PIPELINE DE TREINAMENTO CONCLUÍDO COM SUCESSO!")
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TrustShield Model Training (Orchestrated & Optimized)')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Caminho para o arquivo de configuração (ex: config/config.yaml)'
    )
    args = parser.parse_args()
    train_model(args.config)