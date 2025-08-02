# -*- coding: utf-8 -*-
"""
Componente de Agregação de Dados para Séries Temporais - TrustShield
Versão: 3.0.0 (Enterprise Aprimorada)

Melhorias implementadas na v3.0.0:
- Paralelismo: Adicionado processamento paralelo de batches usando joblib para maior escalabilidade.
- Validação avançada: Checagem de tipos de dados e tratamento de valores ausentes nas features.
- Robustez: Implementado mecanismo de retry para carregamento do modelo do MLflow.
- Monitoramento: Métricas adicionais como distribuição de anomalias e estatísticas descritivas.
- Reprodutibilidade: Adicionado seed para operações aleatórias (se aplicável) e logging de ambiente.
- Modularidade: Métodos subdivididos para melhor legibilidade e testabilidade.
- Configuração: Suporte a argumentos de linha de comando para flexibilidade em orquestração.

Autor: TrustShield Team & IA Gemini
Data: 2025-08-05
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import time
import random

import joblib
import mlflow
import pandas as pd
import sklearn
import yaml
import numpy as np
from joblib import Parallel, delayed

# Adiciona o diretório raiz para importações
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - [TrustShield-Aggregator] - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


LOGGER = get_logger("Aggregator")


class TimeSeriesAggregator:
    def __init__(self, config_path: str = "config/config.yaml", batch_size: int = None):
        LOGGER.info("Inicializando o agregador de séries temporais...")
        self.config = self._load_config(config_path)
        self.model_name = self.config['model_registry']['model_name']
        self.features_path = PROJECT_ROOT / self.config['paths']['data']['featured_dataset']
        self.output_path = PROJECT_ROOT / self.config['paths']['data']['time_series_dataset']
        self.time_column = self.config['time_series']['time_column']
        self.aggregation_freq = self.config['time_series']['aggregation_freq']
        self.batch_size = batch_size or self.config.get('performance', {}).get('batch_size', 100000)
        self.n_jobs = self.config.get('performance', {}).get('n_jobs', -1)  # -1 usa todos os núcleos
        self.retry_attempts = self.config.get('robustness', {}).get('retry_attempts', 3)
        self.random_seed = self.config.get('reproducibility', {}).get('seed', 42)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def _load_config(self, config_path: str) -> dict:
        with open(PROJECT_ROOT / config_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_anomaly_model_with_retry(self):
        for attempt in range(1, self.retry_attempts + 1):
            try:
                LOGGER.info(f"Tentativa {attempt}/{self.retry_attempts}: Carregando o modelo '{self.model_name}' (stage: Staging) do MLflow Registry...")
                model_uri = f"models:/{self.model_name}/Staging"
                return mlflow.pyfunc.load_model(model_uri)
            except Exception as e:
                LOGGER.warning(f"Tentativa {attempt} falhou. Erro: {e}")
                if attempt == self.retry_attempts:
                    LOGGER.error(f"Falha crítica após {self.retry_attempts} tentativas. Verifique o MLflow Registry.")
                    raise
                time.sleep(2 ** attempt)  # Backoff exponencial

    def _validate_data(self, df: pd.DataFrame, model_features: list):
        if df.empty:
            raise ValueError("Dataset de features está vazio. Abortando a execução.")

        if self.time_column not in df.columns:
            raise ValueError(f"Coluna de tempo '{self.time_column}' não encontrada no dataset.")

        missing_cols = set(model_features) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Features esperadas pelo modelo não encontradas no dataset: {missing_cols}")

        # Aprimoramento: Validação de tipos de dados (assumindo features numéricas)
        for col in model_features:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Coluna '{col}' deve ser numérica, mas é do tipo {df[col].dtype}.")

        # Aprimoramento: Tratamento de valores ausentes
        na_count = df[model_features].isna().sum().sum()
        if na_count > 0:
            LOGGER.warning(f"Encontrados {na_count} valores ausentes nas features. Preenchendo com 0 (estratégia simples).")
            df[model_features] = df[model_features].fillna(0)
            mlflow.log_metric("na_filled_count", na_count)

    def _predict_batch(self, batch: pd.DataFrame, model, model_features: list):
        preds = model.predict(batch[model_features])
        return preds['prediction']

    def run(self):
        start_time = time.time()
        LOGGER.info("Iniciando o processo de agregação...")
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

        with mlflow.start_run(run_name=f"Data_Aggregation_for_TS_{time.strftime('%Y%m%d-%H%M')}") as run:
            LOGGER.info(f"MLflow Run ID: {run.info.run_id}")
            mlflow.set_tag("pipeline.type", "data_aggregation_for_ts")

            model = self._load_anomaly_model_with_retry()

            LOGGER.info(f"Carregando dataset de features de {self.features_path}...")
            df_features = pd.read_parquet(self.features_path)
            df_features[self.time_column] = pd.to_datetime(df_features[self.time_column])

            model_features = model.metadata.get_input_schema().input_names()
            self._validate_data(df_features, model_features)
            mlflow.log_metric("missing_features_count", 0)

            LOGGER.info(f"Pontuando {len(df_features)} transações em batches de {self.batch_size} com {self.n_jobs} jobs paralelos...")

            # Aprimoramento: Processamento paralelo de batches
            batches = [df_features.iloc[i:i + self.batch_size] for i in range(0, len(df_features), self.batch_size)]
            all_predictions = Parallel(n_jobs=self.n_jobs)(
                delayed(self._predict_batch)(batch, model, model_features) for batch in batches
            )
            df_features['anomaly'] = np.concatenate(all_predictions)

            # Aprimoramento: Métricas adicionais de monitoramento
            anomaly_rate = df_features['anomaly'].mean()
            anomaly_dist = df_features['anomaly'].value_counts(normalize=True).to_dict()
            date_range_start = df_features[self.time_column].min().isoformat()
            date_range_end = df_features[self.time_column].max().isoformat()
            mlflow.log_metric("historical_anomaly_rate", anomaly_rate)
            mlflow.log_dict(anomaly_dist, "anomaly_distribution.json")
            mlflow.log_param("data_range_start", date_range_start)
            mlflow.log_param("data_range_end", date_range_end)

            LOGGER.info(
                f"Agregando anomalias (taxa: {anomaly_rate:.4f}) por dia (frequência: {self.aggregation_freq})...")
            df_time_series = df_features.set_index(self.time_column).groupby(pd.Grouper(freq=self.aggregation_freq))[
                'anomaly'].sum().reset_index()
            df_time_series.columns = ['ds', 'y']

            # Aprimoramento: Estatísticas descritivas da série temporal
            ts_stats = df_time_series['y'].describe().to_dict()
            mlflow.log_dict(ts_stats, "time_series_stats.json")

            LOGGER.info(f"Série temporal criada com {len(df_time_series)} pontos de dados.")

            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            df_time_series.to_parquet(self.output_path, index=False)
            mlflow.log_artifact(str(self.output_path))

            # Aprimoramento: Reprodutibilidade expandida
            lib_versions = {
                "pandas": pd.__version__,
                "mlflow": mlflow.__version__,
                "sklearn": sklearn.__version__,
                "joblib": joblib.__version__
            }
            mlflow.log_dict(lib_versions, "library_versions.json")
            mlflow.log_param("random_seed", self.random_seed)
            mlflow.log_param("python_version", sys.version)

            runtime = time.time() - start_time
            mlflow.log_metric("runtime_seconds", runtime)
            LOGGER.info(f"✅ Série temporal salva com sucesso em: {self.output_path}")
            LOGGER.info(f"Execução concluída em {runtime:.2f} segundos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agregador de Séries Temporais para TrustShield")
    parser.add_argument("--config", default="config/config.yaml", help="Caminho para o arquivo de configuração")
    parser.add_argument("--batch_size", type=int, help="Tamanho do batch para processamento")
    args = parser.parse_args()

    aggregator = TimeSeriesAggregator(config_path=args.config, batch_size=args.batch_size)
    aggregator.run()
