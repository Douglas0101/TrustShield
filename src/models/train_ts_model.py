# -*- coding: utf-8 -*-
"""
Componente de Treino de Séries Temporais (Forecast) - TrustShield
Versão: 3.0.0 (Enterprise Aprimorada)

Melhorias implementadas na v3.0.0:
- Extensibilidade: Adicionado suporte a múltiplos modelos (Prophet como default, ARIMA opcional via config/args).
- Validação avançada: Checagem de dados (ausentes, tipos) e estratégia de preenchimento.
- Avaliação: Métricas adicionais (MAPE, cobertura de intervalos de confiança).
- Robustez: Mecanismo de retry para carregamento de dados e logging de exceções granulares.
- Reprodutibilidade: Adicionado seed para operações aleatórias e log de ambiente (Python, bibliotecas).
- Visualizações: Gráfico adicional de intervalos de confiança e resíduos aprimorado.
- Modularidade: Métodos subdivididos para treino, previsão e avaliação.
- Configuração: Argumentos expandidos para tipo de modelo e seed.

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

import mlflow
import pandas as pd
import yaml
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from statsmodels.tsa.arima.model import ARIMA  # Para suporte a ARIMA

# Adiciona o diretório raiz para importações
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - [TrustShield-Forecaster] - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


LOGGER = get_logger("Forecaster")


class TimeSeriesForecaster:
    def __init__(self, config_path: str, forecast_end_date: str = None, model_type: str = 'prophet', seed: int = 42):
        LOGGER.info("Inicializando o pipeline de forecast...")
        self.config = self._load_config(config_path)
        self.ts_config = self.config['time_series']
        self.data_path = PROJECT_ROOT / self.config['paths']['data']['time_series_dataset']
        self.model_type = model_type.lower()
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        if forecast_end_date:
            self.ts_config['forecast_horizon_end_date'] = forecast_end_date
            LOGGER.info(f"Data de horizonte de previsão sobrescrita para: {forecast_end_date}")

        if self.model_type not in ['prophet', 'arima']:
            raise ValueError(f"Tipo de modelo '{self.model_type}' não suportado. Opções: 'prophet' ou 'arima'.")

    def _load_config(self, config_path: str) -> dict:
        with open(PROJECT_ROOT / config_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_data_with_retry(self, attempts: int = 3):
        for attempt in range(1, attempts + 1):
            try:
                LOGGER.info(f"Tentativa {attempt}/{attempts}: Carregando série temporal de {self.data_path}...")
                if not self.data_path.exists():
                    raise FileNotFoundError(f"Arquivo de série temporal não encontrado em {self.data_path}.")
                return pd.read_parquet(self.data_path)
            except Exception as e:
                LOGGER.warning(f"Tentativa {attempt} falhou. Erro: {e}")
                if attempt == attempts:
                    LOGGER.error(f"Falha crítica após {attempts} tentativas.")
                    raise
                time.sleep(2 ** attempt)

    def _validate_data(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError("Série temporal está vazia. Abortando a execução.")

        required_cols = ['ds', 'y']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Colunas requeridas ausentes: {missing_cols}")

        df['ds'] = pd.to_datetime(df['ds'])
        if not pd.api.types.is_numeric_dtype(df['y']):
            raise ValueError("Coluna 'y' deve ser numérica.")

        na_count = df.isna().sum().sum()
        if na_count > 0:
            LOGGER.warning(f"Encontrados {na_count} valores ausentes. Preenchendo com interpolação linear.")
            df = df.interpolate(method='linear')
            mlflow.log_metric("na_filled_count", na_count)
        return df

    def _train_model(self, train_df: pd.DataFrame):
        if self.model_type == 'prophet':
            params = self.ts_config.get('prophet_params', {})
            model = Prophet(**params)
            model.fit(train_df)
        elif self.model_type == 'arima':
            params = self.ts_config.get('arima_params', {'order': (5, 1, 0)})
            model = ARIMA(train_df['y'], order=params['order'], dates=train_df['ds'])
            model = model.fit()
        return model

    def _predict(self, model, future_df: pd.DataFrame):
        if self.model_type == 'prophet':
            return model.predict(future_df)
        elif self.model_type == 'arima':
            forecast_steps = len(future_df) - len(model.data.orig_endog)
            forecast = model.forecast(steps=forecast_steps)
            return pd.DataFrame({'ds': future_df['ds'], 'yhat': forecast})

    def _evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, forecast: pd.DataFrame = None):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.nan

        metrics = {"validation_mae": mae, "validation_rmse": rmse, "validation_mape": mape}

        if forecast is not None and 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
            coverage = np.mean((y_true >= forecast['yhat_lower']) & (y_true <= forecast['yhat_upper']))
            metrics["confidence_interval_coverage"] = coverage

        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        LOGGER.info(f"Métricas de validação: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.4f}%")
        return metrics

    def run(self):
        start_time = time.time()
        LOGGER.info("Iniciando o processo de treino do modelo de forecast...")
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

        with mlflow.start_run(run_name=f"TimeSeries_Forecast_{time.strftime('%Y%m%d-%H%M')}") as run:
            mlflow.set_tag("pipeline.type", "time_series_forecasting")
            LOGGER.info(f"MLflow Run ID: {run.info.run_id}")
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("random_seed", self.seed)

            df_ts = self._load_data_with_retry()
            df_ts = self._validate_data(df_ts)

            train_size = int(len(df_ts) * 0.8)
            train_df, val_df = df_ts.iloc[:train_size], df_ts.iloc[train_size:]
            LOGGER.info(f"Dividindo dados: {len(train_df)} para treino, {len(val_df)} para validação.")

            LOGGER.info(f"Treinando modelo {self.model_type} com dados de treino...")
            model = self._train_model(train_df)

            LOGGER.info("Avaliando o modelo com dados de validação...")
            future_val = pd.DataFrame({'ds': pd.date_range(start=df_ts['ds'].min(), periods=len(df_ts), freq=self.ts_config['aggregation_freq'])})
            forecast_val = self._predict(model, future_val)

            y_true = val_df['y'].values
            y_pred = forecast_val['yhat'][-len(val_df):].values
            self._evaluate(y_true, y_pred, forecast_val.iloc[-len(val_df):] if self.model_type == 'prophet' else None)

            LOGGER.info(f"Retreinando o modelo {self.model_type} com o dataset completo...")
            full_model = self._train_model(df_ts)

            end_date = pd.to_datetime(self.ts_config['forecast_horizon_end_date'])
            periods = (end_date - df_ts['ds'].max()).days + 1  # +1 para incluir o dia final
            LOGGER.info(f"Gerando previsão para {periods} períodos até {end_date.date()}...")

            future_full = pd.DataFrame({'ds': pd.date_range(start=df_ts['ds'].min(), periods=len(df_ts) + periods, freq=self.ts_config['aggregation_freq'])})
            forecast_full = self._predict(full_model, future_full)

            LOGGER.info("Gerando e salvando artefatos gráficos e de dados...")
            if self.model_type == 'prophet':
                fig_forecast = plot_plotly(full_model, forecast_full)
                fig_components = plot_components_plotly(full_model, forecast_full)
                fig_forecast.write_html("forecast_plot.html")
                fig_components.write_html("components_plot.html")
                mlflow.log_artifact("forecast_plot.html")
                mlflow.log_artifact("components_plot.html")

            # Gráfico de Resíduos (comum a ambos)
            df_residuals = pd.DataFrame({'ds': val_df['ds'], 'residuals': y_true - y_pred})
            fig_residuals = go.Figure([go.Scatter(x=df_residuals['ds'], y=df_residuals['residuals'], mode='markers', name='Resíduos')])
            fig_residuals.update_layout(title='Resíduos do Modelo (Validação)', xaxis_title='Data', yaxis_title='Erro')
            fig_residuals.write_html("residuals_plot.html")
            mlflow.log_artifact("residuals_plot.html")

            # Gráfico de Intervalos de Confiança (se disponível)
            if 'yhat_lower' in forecast_full.columns:
                fig_ci = go.Figure([
                    go.Scatter(x=forecast_full['ds'], y=forecast_full['yhat'], name='Previsão'),
                    go.Scatter(x=forecast_full['ds'], y=forecast_full['yhat_upper'], fill=None, mode='lines', line_color='lightgrey', name='Upper CI'),
                    go.Scatter(x=forecast_full['ds'], y=forecast_full['yhat_lower'], fill='tonexty', mode='lines', line_color='lightgrey', name='Lower CI')
                ])
                fig_ci.update_layout(title='Previsão com Intervalos de Confiança', xaxis_title='Data', yaxis_title='Valor')
                fig_ci.write_html("confidence_intervals.html")
                mlflow.log_artifact("confidence_intervals.html")

            forecast_full.to_parquet("full_forecast.parquet")
            mlflow.log_artifact("full_forecast.parquet")

            LOGGER.info(f"Registrando o modelo {self.model_type} final no MLflow...")
            if self.model_type == 'prophet':
                mlflow.prophet.log_model(full_model, artifact_path="prophet-model")
            elif self.model_type == 'arima':
                mlflow.statsmodels.log_model(full_model, artifact_path="arima-model")

            # Reprodutibilidade expandida
            lib_versions = {"pandas": pd.__version__, "mlflow": mlflow.__version__, "prophet": Prophet.__version__ if self.model_type == 'prophet' else "N/A"}
            mlflow.log_dict(lib_versions, "library_versions.json")
            mlflow.log_param("python_version", sys.version)

            runtime = time.time() - start_time
            mlflow.log_metric("runtime_seconds", runtime)
            LOGGER.info(f"✅ Pipeline de forecast concluído com sucesso em {runtime:.2f} segundos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Treino de Modelo de Forecast - TrustShield")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Caminho para o arquivo de configuração.")
    parser.add_argument("--end-date", type=str, help="Sobrescreve a data final da previsão (formato: YYYY-MM-DD).")
    parser.add_argument("--model-type", type=str, default="prophet", help="Tipo de modelo: 'prophet' ou 'arima'.")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade.")
    args = parser.parse_args()

    forecaster = TimeSeriesForecaster(config_path=args.config, forecast_end_date=args.end_date, model_type=args.model_type, seed=args.seed)
    forecaster.run()
