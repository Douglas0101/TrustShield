# -*- coding: utf-8 -*-
"""
Sistema de Treinamento TrustShield - ARQUITETURA ENTERPRISE (Otimizado)
Versão: 10.3.0-final

Principais otimizações:
- Corrigido o acesso ao número da versão do modelo para ser compatível com a API do MLflow no ambiente.
- HPO com contamination fixo, objetivo mais informativo e TPE seed.
- Split temporal opcional e validações robustas de schema e config.
- Calibração de threshold por percentil dos scores de validação.
- Servir anomaly_score contínuo + predição binária.
- Assinatura/artefatos MLflow aprimorados e guardrails de promoção.

Autor: TrustShield Team & IA Gemini
Data: 2025-08-03
"""

import argparse
import gc
import hashlib
import logging
import os
import random
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlflow
import numpy as np
import optuna
import pandas as pd
import pandera as pa
import sklearn
import yaml
from mlflow.models.signature import ModelSignature, infer_signature
from scipy.stats import ks_2samp
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings('default')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================================================
# 🏗️ CONFIGURAÇÃO E SETUP INICIAL
# =====================================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - [TrustShield] - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


LOGGER = get_logger("Trainer")


# =====================================================================================
# 🏛️ COMPONENTES DA ARQUITETURA
# =====================================================================================

@dataclass
class Artifacts:
    model: Any
    scaler: Any
    metrics: Dict[str, float]
    params: Dict[str, Any]
    signature: ModelSignature
    threshold: float
    feature_names: List[str]


class ConfigValidator:
    """Validações básicas no arquivo de configuração."""

    @staticmethod
    def validate(config: Dict[str, Any]) -> None:
        required_keys = [
            ('paths', 'data', 'featured_dataset'),
            ('models', 'isolation_forest', 'features'),
            ('project', 'random_state'),
            ('training', 'test_size'),
            ('training', 'validation_size'),
            ('mlflow', 'experiment_name'),
            ('hyper_optimization', 'n_trials'),
        ]
        for path in required_keys:
            node = config
            for k in path:
                if k not in node:
                    raise ValueError(f"Config inválida: chave ausente {'.'.join(path)}")
                node = node[k]

        ts = config['training']['test_size']
        vs = config['training']['validation_size']
        if not (0 < ts < 1 and 0 < vs < 1 and (ts + vs) < 1):
            raise ValueError("training.test_size e training.validation_size devem estar em (0,1) e somar < 1.")

        hpo_space = config.get('hyper_optimization', {}).get('space', {})
        hpo_space.setdefault('n_estimators', [100, 600])
        hpo_space.setdefault('max_samples', [0.5, 1.0])
        hpo_space.setdefault('max_features', [0.5, 1.0])
        config['hyper_optimization']['space'] = hpo_space

        if 'fixed_contamination' not in config['hyper_optimization']:
            config['hyper_optimization']['fixed_contamination'] = 0.01

        if 'scaler' not in config.get('preprocessing', {}):
            config.setdefault('preprocessing', {}).setdefault('scaler', 'standard')

        config.setdefault('preprocessing', {}).setdefault('use_time_split', False)
        time_col = config.get('preprocessing', {}).get('time_column')
        if config['preprocessing']['use_time_split'] and not time_col:
            raise ValueError("use_time_split=True requer preprocessing.time_column definido.")


class DataPipeline:
    """Ingestão, validação, split e checks de drift."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_path = PROJECT_ROOT / config['paths']['data']['featured_dataset']
        self.model_features = config['models']['isolation_forest']['features']
        self.schema = pa.DataFrameSchema({
            col: pa.Column(
                float,
                required=True,
                coerce=True,
                checks=[
                    pa.Check(lambda s: s.notna().all(), error="NaN encontrado"),
                    pa.Check(lambda s: np.isfinite(s).all(), error="Valores não finitos (inf, -inf) encontrados")
                ]
            )
            for col in self.model_features
        })

    def get_data_hash(self, df: pd.DataFrame) -> str:
        return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

    def _read(self) -> pd.DataFrame:
        LOGGER.info(f"Lendo dados de {self.data_path} ...")
        return pd.read_parquet(self.data_path)

    def _sample(self, df: pd.DataFrame) -> pd.DataFrame:
        frac = self.config.get('preprocessing', {}).get('sample_frac', 1.0)
        if 0 < frac < 1.0:
            LOGGER.warning(f"Utilizando uma amostra de {frac * 100:.1f}% dos dados.")
            df = df.sample(frac=frac, random_state=self.config['project']['random_state'])
        return df

    def _split(self, df_model: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        test_size = self.config['training']['test_size']
        val_size = self.config['training']['validation_size']
        random_state = self.config['project']['random_state']
        use_time_split = self.config['preprocessing']['use_time_split']
        time_col = self.config.get('preprocessing', {}).get('time_column')

        if use_time_split:
            if time_col not in df_model.columns:
                raise ValueError(f"Coluna temporal '{time_col}' não está presente nos dados.")
            LOGGER.info(f"Executando split temporal via coluna '{time_col}'...")
            df_sorted = df_model.sort_values(time_col)
            n = len(df_sorted)
            train_end = int(n * (1 - (test_size + val_size)))
            val_end = int(n * (1 - test_size))
            df_train = df_sorted.iloc[:train_end].drop(columns=[time_col])
            df_val = df_sorted.iloc[train_end:val_end].drop(columns=[time_col])
            df_test = df_sorted.iloc[val_end:].drop(columns=[time_col])
        else:
            LOGGER.info("Executando split aleatório (não temporal)...")
            if time_col and time_col in df_model.columns:
                df_model = df_model.drop(columns=[time_col])
            df_train, df_temp = train_test_split(
                df_model, test_size=(test_size + val_size), random_state=random_state, shuffle=True
            )
            df_val, df_test = train_test_split(
                df_temp, test_size=(test_size / (test_size + val_size)), random_state=random_state, shuffle=True
            )

        return df_train, df_val, df_test

    def _drift_checks(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict[str, float]:
        LOGGER.info("Calculando métricas de drift (KS) entre treino e teste...")
        ks_metrics = {}
        for col in df_train.columns:
            try:
                ks = ks_2samp(df_train[col].astype(float), df_test[col].astype(float)).statistic
                ks_metrics[f"drift_ks_{col}"] = float(ks)
            except Exception as e:
                LOGGER.warning(f"KS falhou para coluna {col}: {e}")
        return ks_metrics

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
        LOGGER.info("Iniciando pipeline de dados...")
        df = self._read()
        df = self._sample(df)

        required_cols = list(self.model_features)
        time_col = self.config.get('preprocessing', {}).get('time_column')
        if self.config['preprocessing']['use_time_split'] and time_col:
            required_cols = required_cols + [time_col]

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Colunas ausentes no dataset: {missing}")

        df_model = df[required_cols].copy()

        LOGGER.info("Validando schema dos dados de features...")
        if time_col and time_col in df_model.columns:
            self.schema.validate(df_model.drop(columns=[time_col]))
        else:
            self.schema.validate(df_model)

        df_train, df_val, df_test = self._split(df_model)
        LOGGER.info(f"Dados divididos: {len(df_train)} treino, {len(df_val)} validação, {len(df_test)} teste.")

        drift_metrics = self._drift_checks(df_train, df_test)
        return df_train, df_val, df_test, drift_metrics


class HyperparameterTuner:
    """Otimização de hiperparâmetros com objetivo mais informativo e contamination fixo."""

    def __init__(self, config: Dict[str, Any], df_train: pd.DataFrame, df_val: pd.DataFrame):
        self.config = config
        self.df_train = df_train
        self.df_val = df_val
        self.hpo_config = config['hyper_optimization']

    def _get_scaler(self):
        scaler_type = self.config.get('preprocessing', {}).get('scaler', 'standard').lower()
        if scaler_type == 'robust':
            return RobustScaler()
        return StandardScaler()

    def _objective(self, trial: optuna.trial.Trial) -> float:
        with mlflow.start_run(nested=True):
            mlflow.set_tag("mlflow.runName", f"hpo-trial-{trial.number}")

            space = self.hpo_config['space']
            params = {
                'n_estimators': trial.suggest_int('n_estimators', space['n_estimators'][0], space['n_estimators'][1]),
                'max_samples': trial.suggest_float('max_samples', space['max_samples'][0], space['max_samples'][1]),
                'max_features': trial.suggest_float('max_features', space['max_features'][0], space['max_features'][1]),
                'contamination': float(self.hpo_config.get('fixed_contamination', 0.01)),
                'n_jobs': -1,
                'random_state': self.config['project']['random_state'],
                'bootstrap': False,
                'warm_start': False
            }

            params['max_features'] = min(max(params['max_features'], 0.5), 1.0)
            params['max_samples'] = min(max(params['max_samples'], 0.1), 1.0)

            mlflow.log_params(params)

            scaler = self._get_scaler()
            X_train_scaled = scaler.fit_transform(self.df_train)
            X_val_scaled = scaler.transform(self.df_val)

            model = IsolationForest(**params)
            t0 = time.time()
            model.fit(X_train_scaled)
            train_time = time.time() - t0

            scores_val = -model.decision_function(X_val_scaled)
            anomaly_rate = float(np.mean(scores_val >= np.quantile(scores_val, 1 - params['contamination'])))
            score_var = float(np.var(scores_val))
            alpha = 0.1
            objective = -(score_var) + alpha * abs(anomaly_rate - params['contamination'])

            mlflow.log_metric("val_anomaly_rate_proxy", anomaly_rate)
            mlflow.log_metric("val_score_variance", score_var)
            mlflow.log_metric("train_time_sec", train_time)
            mlflow.log_metric("objective", objective)

            return objective

    def run(self) -> Dict[str, Any]:
        LOGGER.info(f"Iniciando otimização de hiperparâmetros com {self.hpo_config['n_trials']} tentativas...")
        sampler = optuna.samplers.TPESampler(seed=self.config['project']['random_state'])
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
        study.optimize(self._objective, n_trials=self.hpo_config['n_trials'],
                       timeout=self.hpo_config.get('timeout_sec'))

        LOGGER.info(f"Otimização concluída. Melhor valor (objetivo): {study.best_value:.6f}")
        LOGGER.info(f"Melhores hiperparâmetros encontrados: {study.best_params}")
        return study.best_params


class ModelTrainer:
    """Treinando modelo final e calibrando threshold."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _get_scaler(self):
        scaler_type = self.config.get('preprocessing', {}).get('scaler', 'standard').lower()
        if scaler_type == 'robust':
            return RobustScaler()
        return StandardScaler()

    def run(self, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame,
            best_params: Dict[str, Any]) -> Artifacts:
        LOGGER.info("Treinando modelo final com os melhores hiperparâmetros...")

        scaler = self._get_scaler()

        X_train = pd.concat([df_train, df_val])
        X_test = df_test.copy()

        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(df_val)
        X_test_scaled = scaler.transform(X_test)

        params = dict(best_params)
        params.update({
            'contamination': float(self.config['hyper_optimization'].get('fixed_contamination', 0.01)),
            'n_jobs': -1,
            'random_state': self.config['project']['random_state'],
            'bootstrap': False,
            'warm_start': False
        })

        model = IsolationForest(**params)

        t0 = time.time()
        model.fit(X_train_scaled)
        training_time = time.time() - t0

        scores_val = -model.decision_function(X_val_scaled)
        desired_percentile = 100 - (params['contamination'] * 100.0)
        threshold = float(np.percentile(scores_val, desired_percentile))

        t1 = time.time()
        scores_test = -model.decision_function(X_test_scaled)
        infer_latency = (time.time() - t1) / max(1, len(X_test))

        preds_test = (scores_test >= threshold).astype(int)
        anomaly_rate_test = float(np.mean(preds_test == 1))

        metrics = {
            "test_anomaly_rate": anomaly_rate_test,
            "training_time_sec": float(training_time),
            "inference_latency_s_per_row": float(infer_latency),
            "threshold_calibrated": threshold,
        }
        LOGGER.info(f"Métricas no conjunto de teste: {metrics}")

        pred_df_example = pd.DataFrame({
            "anomaly_score": scores_test[:5],
            "prediction": preds_test[:5]
        })
        signature = infer_signature(X_train.iloc[:5], pred_df_example)

        return Artifacts(
            model=model,
            scaler=scaler,
            metrics=metrics,
            params=params,
            signature=signature,
            threshold=threshold,
            feature_names=list(X_train.columns)
        )


class ModelPromoter:
    """Registro e promoção com guardrails no MLflow Model Registry."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry_config = config['model_registry']
        self.model_name = self.registry_config['model_name']

    def run(self, artifacts: Artifacts, data_hashes: Dict[str, str], git_hash: str, input_example: pd.DataFrame,
            extra_metrics: Dict[str, float]):
        LOGGER.info(f"Registrando modelo '{self.model_name}' no MLflow Model Registry...")

        class ModelWrapper(mlflow.pyfunc.PythonModel):
            def __init__(self, model, scaler, threshold, feature_names):
                self.model = model
                self.scaler = scaler
                self.threshold = threshold
                self.feature_names = feature_names

            def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
                if list(model_input.columns) != self.feature_names:
                    raise ValueError("Colunas de entrada divergentes do treino.")
                X = self.scaler.transform(model_input)
                scores = -self.model.decision_function(X)
                pred = (scores >= self.threshold).astype(int)
                return pd.DataFrame({"anomaly_score": scores, "prediction": pred})

        pyfunc_model = ModelWrapper(artifacts.model, artifacts.scaler, artifacts.threshold, artifacts.feature_names)

        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=pyfunc_model,
            registered_model_name=self.model_name,
            signature=artifacts.signature,
            input_example=input_example.head(5),
        )

        client = mlflow.tracking.MlflowClient()

        # --- CORREÇÃO APLICADA AQUI ---
        # Acessa a versão do modelo diretamente do objeto `registered_model_version`.
        # O log de erro indica que este objeto é uma string, então não usamos .version.
        if hasattr(model_info, 'registered_model_version') and model_info.registered_model_version:
             # Para versões mais recentes do MLflow, o objeto ModelVersion está aqui
             if hasattr(model_info.registered_model_version, 'version'):
                 model_version = model_info.registered_model_version.version
             # Para a versão no ambiente do usuário, o log indica que é uma string
             else:
                 model_version = str(model_info.registered_model_version)
        else:
            # Fallback para versões mais antigas do MLflow
            latest_versions = client.get_latest_versions(self.model_name, stages=["None"])
            if not latest_versions:
                raise RuntimeError("Não foi possível encontrar a nova versão do modelo no registro.")
            model_version = latest_versions[0].version


        client.set_model_version_tag(name=self.model_name, version=model_version, key="data_hash_train",
                                     value=data_hashes["train"])
        client.set_model_version_tag(name=self.model_name, version=model_version, key="data_hash_val",
                                     value=data_hashes["val"])
        client.set_model_version_tag(name=self.model_name, version=model_version, key="data_hash_test",
                                     value=data_hashes["test"])
        client.set_model_version_tag(name=self.model_name, version=model_version, key="git_hash", value=git_hash)
        client.set_model_version_tag(name=self.model_name, version=model_version, key="features",
                                     value=",".join(artifacts.feature_names))
        client.set_model_version_tag(name=self.model_name, version=model_version, key="scaler_type",
                                     value=type(artifacts.scaler).__name__)
        client.set_model_version_tag(name=self.model_name, version=model_version, key="threshold",
                                     value=str(artifacts.threshold))

        threshold_metric = self.registry_config['promotion_threshold']
        latency_guard = self.registry_config.get('latency_p95_guard_s_per_row', 0.1)

        guard_ok = (artifacts.metrics['test_anomaly_rate'] <= threshold_metric) and \
                   (artifacts.metrics['inference_latency_s_per_row'] <= latency_guard)

        mlflow.log_metric("guardrail_passing", int(guard_ok))
        for k, v in extra_metrics.items():
            mlflow.log_metric(k, float(v))

        if guard_ok:
            LOGGER.info(
                f"Guardrails atendidos (test_anomaly_rate={artifacts.metrics['test_anomaly_rate']:.4f} <= {threshold_metric}, "
                f"latency_s/row={artifacts.metrics['inference_latency_s_per_row']:.6f} <= {latency_guard}). "
                f"Promovendo para 'Staging'..."
            )
            client.transition_model_version_stage(
                name=self.model_name,
                version=model_version,
                stage="Staging",
                archive_existing_versions=True
            )
            LOGGER.info("Modelo promovido com sucesso.")
        else:
            LOGGER.warning("Guardrails não atendidos. Modelo permanece em 'None'.")


# =====================================================================================
# 🎼 ORQUESTRADOR PRINCIPAL DO PIPELINE
# =====================================================================================

class TrainingPipeline:
    """Orquestra a execução de todos os componentes da arquitetura de treino."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        ConfigValidator.validate(self.config)
        self._setup_mlflow()
        self._seed_everything(self.config['project']['random_state'])
        self.data_pipeline = DataPipeline(self.config)
        self.model_promoter = ModelPromoter(self.config)
        self.trainer = ModelTrainer(self.config)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(PROJECT_ROOT / config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_mlflow(self):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

    def _seed_everything(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)

    def _get_git_hash(self) -> str:
        try:
            return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        except Exception:
            LOGGER.warning("Não foi possível obter o hash do Git. O repositório não é Git ou o executável não está no PATH.")
            return "N/A"

    def _log_env_versions(self):
        env = {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "optuna": optuna.__version__,
            "mlflow": mlflow.__version__
        }
        mlflow.log_dict(env, "env_versions.json")

    def run(self):
        start_time = time.time()

        df_train, df_val, df_test, drift_metrics = self.data_pipeline.run()
        data_hash_train = self.data_pipeline.get_data_hash(df_train)
        data_hash_val = self.data_pipeline.get_data_hash(df_val)
        data_hash_test = self.data_pipeline.get_data_hash(df_test)
        git_hash = self._get_git_hash()

        with mlflow.start_run(run_name=f"HPO_Study_{datetime.now():%Y%m%d-%H%M}"):
            run_id = mlflow.active_run().info.run_id
            LOGGER.info(f"RunID: {run_id}")

            mlflow.set_tags({
                "pipeline.type": "training_and_hpo",
                "data.train_hash": data_hash_train,
                "data.val_hash": data_hash_val,
                "data.test_hash": data_hash_test,
                "code.git_hash": git_hash,
                "data.path": str(self.data_pipeline.data_path),
                "features": ",".join(self.data_pipeline.model_features)
            })

            self._log_env_versions()
            for k, v in drift_metrics.items():
                mlflow.log_metric(k, v)

            tuner = HyperparameterTuner(self.config, df_train, df_val)
            best_params = tuner.run()
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

            final_artifacts = self.trainer.run(df_train, df_val, df_test, best_params)
            mlflow.log_metrics(final_artifacts.metrics)
            mlflow.log_params({
                "decision_threshold": final_artifacts.threshold,
                "scaler_type": type(final_artifacts.scaler).__name__
            })
            mlflow.log_dict({"feature_names": final_artifacts.feature_names}, "feature_names.json")

            input_example = df_train.head(5).copy()

            self.model_promoter.run(
                final_artifacts,
                data_hashes={"train": data_hash_train, "val": data_hash_val, "test": data_hash_test},
                git_hash=git_hash,
                input_example=input_example,
                extra_metrics=drift_metrics
            )

        total_time = time.time() - start_time
        LOGGER.info(f"🎉 PIPELINE DE TREINO ENTERPRISE CONCLUÍDO com sucesso em {total_time:.2f} segundos.")
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Pipeline de Treino Enterprise - TrustShield (Otimizado)")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Caminho para o arquivo de configuração.")
    parser.add_argument("--model", type=str, help="Argumento ignorado.")
    args = parser.parse_args()
    try:
        pipeline = TrainingPipeline(config_path=args.config)
        pipeline.run()
    except Exception as e:
        LOGGER.exception(f"❌ ERRO CRÍTICO no pipeline de treino: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
