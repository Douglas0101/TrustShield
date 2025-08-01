# -*- coding: utf-8 -*-
"""
Módulo de Inferência de Produção - Projeto TrustShield
Versão: 4.1.0-robust-paths

Este módulo representa o motor de inferência final do TrustShield.

🎯 Funcionalidades Principais:
1.  ✅ Carregamento de Artefatos: Lida com os artefatos (modelo + scaler).
2.  ✅ Match Perfeito de Features: Alinha os dados de entrada com o schema exato do modelo.
3.  ✅ Performance Otimizada: Configurado para o hardware alvo.
4.  ✅ API-Ready: Estruturado para ser envolvido por uma API como a FastAPI.
5.  ✅ Monitoramento e Logging: Inclui monitoramento para observabilidade.

Autor: IA Gemini com base na arquitetura TrustShield
Data: 2025-08-01
"""

import logging
import os
import psutil
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import joblib
import numpy as np
import pandas as pd

# Configurações de otimização de performance
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['MKL_DYNAMIC'] = 'FALSE'

warnings.filterwarnings('ignore')


class ResourceMonitor:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.process = psutil.Process()
        self.start_time = time.time()
        self.prediction_count = 0
        self.total_inference_time = 0
        self.success_count = 0

    def get_current_stats(self) -> Dict[str, Any]:
        memory = psutil.virtual_memory()
        return {
            'cpu_usage_percent': round(psutil.cpu_percent(interval=0.1), 1),
            'memory_usage_percent': round(memory.percent, 1),
            'avg_inference_time_ms': round((self.total_inference_time / max(self.prediction_count, 1)) * 1000, 2),
            'uptime_seconds': round(time.time() - self.start_time, 1)
        }

    def log_prediction_stats(self, inference_time: float, batch_size: int = 1, success: bool = True):
        self.prediction_count += batch_size
        self.total_inference_time += inference_time
        if success: self.success_count += batch_size
        throughput = batch_size / inference_time if inference_time > 0 else float('inf')
        status_icon = "✅" if success else "❌"
        self.logger.info(
            f"📊 {status_icon} Predição: {batch_size} amostra(s) em {inference_time * 1000:.1f}ms | Throughput: {throughput:.0f} amostras/s")


class TrustShieldPredictor:
    def __init__(self, model_path: Optional[Path] = None):
        self.logger = self._setup_logger()
        self.monitor = ResourceMonitor(self.logger)
        try:
            self.model_path = model_path or self._auto_detect_paths()
            self._load_artifacts()
            self.logger.info("🎯 Motor de Inferência TrustShield inicializado com sucesso!")
        except Exception as e:
            self.logger.critical(f"❌ Erro crítico na inicialização do preditor: {e}", exc_info=True)
            raise

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('TrustShieldPredictor')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            formatter = logging.Formatter('%(asctime)s - [TrustShield-Predictor] - %(levelname)s - %(message)s')
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _auto_detect_paths(self) -> Path:
        # CORREÇÃO: Constrói o caminho de forma robusta a partir da localização do ficheiro.
        project_root = Path(__file__).resolve().parents[2]
        model_dir = project_root / "outputs" / "models"

        if not model_dir.exists():
            raise FileNotFoundError(f"Diretório de modelos não encontrado: {model_dir}")

        # Encontra o ficheiro de modelo .joblib mais recente.
        latest_model = max(model_dir.glob("*.joblib"), key=lambda p: p.stat().st_mtime, default=None)

        if not latest_model:
            raise FileNotFoundError(f"Nenhum modelo .joblib encontrado no diretório: {model_dir}")

        self.logger.info(f"📁 Modelo mais recente detectado: {latest_model.name}")
        return latest_model

    def _load_artifacts(self):
        start_time = time.time()
        self.logger.info(f"📥 Carregando artefatos de: {self.model_path}")
        artifacts = joblib.load(self.model_path)

        self.model = artifacts.get('model')
        self.scaler = artifacts.get('scaler')

        if not self.model: raise ValueError("O artefato não contém um modelo válido.")

        self.model_features = self._extract_model_features()
        self.model_type = self.model.__class__.__name__
        self.logger.info(f"✅ Artefatos carregados em {time.time() - start_time:.2f}s | Tipo: {self.model_type}")

    def _extract_model_features(self) -> List[str]:
        feature_names = getattr(self.model, 'feature_names_in_', None) or getattr(self.scaler, 'feature_names_in_',
                                                                                  None)
        if feature_names is not None: return list(feature_names)
        raise AttributeError("O artefato do modelo não contém a lista de features ('feature_names_in_').")

    def _prepare_input_data(self, transaction_data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        df = pd.DataFrame([transaction_data]) if isinstance(transaction_data, dict) else transaction_data.copy()

        categorical_cols = {'gender', 'use_chip'}
        for col in categorical_cols:
            if col in df.columns:
                df = pd.get_dummies(df, columns=[col], prefix=col, dtype='int8')

        final_df = pd.DataFrame(0, index=df.index, columns=self.model_features, dtype='float32')
        common_cols = df.columns.intersection(self.model_features)
        final_df[common_cols] = df[common_cols]

        if self.scaler:
            scaled_data = self.scaler.transform(final_df)
            final_df = pd.DataFrame(scaled_data, columns=self.model_features, index=final_df.index, dtype='float32')

        return final_df

    def predict(self, transaction_data: Union[Dict, pd.DataFrame]) -> Dict[str, Any]:
        start_time = time.time()
        try:
            prepared_df = self._prepare_input_data(transaction_data)
            predictions = self._execute_prediction(prepared_df)
            confidence_scores = self._calculate_confidence(prepared_df)
            inference_time = time.time() - start_time
            self.monitor.log_prediction_stats(inference_time, len(prepared_df))

            is_single = isinstance(transaction_data, dict)
            result_prediction = int(predictions[0]) if is_single else [int(p) for p in predictions]
            result_confidence = float(confidence_scores[0]) if is_single else [float(c) for c in confidence_scores]

            return {
                'prediction': result_prediction,
                'prediction_label': 'ANOMALIA' if result_prediction == -1 else 'NORMAL',
                'confidence_score': result_confidence,
                'inference_time_ms': round(inference_time * 1000, 2),
                'success': True
            }
        except Exception as e:
            self.monitor.log_prediction_stats(time.time() - start_time, 1, success=False)
            self.logger.error(f"❌ Falha durante a predição: {e}", exc_info=True)
            return {'prediction': 1, 'prediction_label': 'NORMAL', 'error': str(e), 'success': False}

    def _execute_prediction(self, prepared_df: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, 'n_jobs'): self.model.n_jobs = 4

        if hasattr(self.model, 'predict'):
            return self.model.predict(prepared_df)
        elif hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(prepared_df)
            return np.where(scores >= 0, 1, -1)
        raise NotImplementedError(f"O modelo {self.model_type} não tem método 'predict' ou 'decision_function'.")

    def _calculate_confidence(self, prepared_df: pd.DataFrame) -> np.ndarray:
        try:
            if hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(prepared_df)
                return 1 / (1 + np.exp(-np.abs(scores)))
        except Exception as e:
            self.logger.warning(f"⚠️ Não foi possível calcular o score de confiança: {e}")
        return np.full(len(prepared_df), 0.5)

    def get_status(self) -> Dict[str, Any]:
        return {
            'status': 'OPERATIONAL',
            'model_info': {'type': self.model_type, 'path': str(self.model_path.name)},
            'performance_metrics': self.monitor.get_current_stats()
        }


def run_demo():
    print("\n" + "=" * 80 + "\n🚀 DEMONSTRAÇÃO DO MOTOR DE INFERÊNCIA TRUSTSHIELD\n" + "=" * 80)
    try:
        predictor = TrustShieldPredictor()
        status = predictor.get_status()
        print(
            f"\n📊 STATUS INICIAL:\n  ● Modelo: {status['model_info']['type']}\n  ● Caminho: {status['model_info']['path']}")

        suspicious_transaction = {'amount': 8750.00, 'transaction_hour': 2, 'credit_score': 510, 'amount_vs_avg': 50.0}
        print("\n" + "-" * 80 + "\n🚨 EXEMPLO 1: Transação de Alto Risco")
        result = predictor.predict(suspicious_transaction)
        print(f"  ▶️ Resultado: {result['prediction_label']} (Score: {result.get('confidence_score', 0):.3f})")

        normal_transaction = {'amount': 45.75, 'transaction_hour': 14, 'credit_score': 780, 'amount_vs_avg': 0.8}
        print("\n" + "-" * 80 + "\n✅ EXEMPLO 2: Transação de Baixo Risco")
        result = predictor.predict(normal_transaction)
        print(f"  ▶️ Resultado: {result['prediction_label']} (Score: {result.get('confidence_score', 0):.3f})")

        print("\n" + "=" * 80)

    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO DURANTE A DEMONSTRAÇÃO: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_demo()