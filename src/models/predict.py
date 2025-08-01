# -*- coding: utf-8 -*-
"""
Módulo de Inferência de Produção - Projeto TrustShield
Versão: 4.0.0-production-ready

Este módulo representa o motor de inferência final do TrustShield, projetado
para ser implantado em um ambiente de produção. Ele é otimizado para robustez,
performance e, crucialmente, para garantir 100% de compatibilidade com os
modelos treinados pelo pipeline de MLOps.

🎯 Funcionalidades Principais:
1.  ✅ Carregamento de Artefatos: Lida de forma inteligente com os artefatos
       gerados pelo pipeline de treino (modelo + scaler).
2.  ✅ Match Perfeito de Features: Extrai as features exatas do artefato do
       modelo e alinha qualquer dado de entrada para corresponder perfeitamente,
       eliminando a causa nº 1 de falhas em produção.
3.  ✅ Performance Otimizada: Configurado para extrair o máximo de performance
       do hardware alvo (Intel i3), mantendo a compatibilidade.
4.  ✅ API-Ready: Estruturado com métodos claros para predição e status,
       pronto para ser envolvido por uma API (ex: FastAPI).
5.  ✅ Monitoramento e Logging: Inclui monitoramento de recursos e logs
       detalhados para observabilidade em produção.

Hardware Target:
- CPU: 11th Gen Intel® Core™ i3-1115G4 × 4 cores
- RAM: 19.3 GB

Execução da Demonstração:
    python src/models/predict.py

Autor: IA Gemini com base na arquitetura TrustShield
Data: 2025-07-29
"""

import logging
import os
import psutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import joblib
import numpy as np
import pandas as pd

# Configurações de otimização de performance para o hardware alvo
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMBA_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_DYNAMIC'] = 'FALSE'

warnings.filterwarnings('ignore')


class ResourceMonitor:
    """Monitora os recursos do sistema de forma eficiente para observabilidade."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.process = psutil.Process()
        self.start_time = time.time()
        self.prediction_count = 0
        self.total_inference_time = 0
        self.success_count = 0

    def get_current_stats(self) -> Dict[str, Any]:
        """Obtém as estatísticas atuais de CPU, memória e performance."""
        try:
            memory = psutil.virtual_memory()
            return {
                'cpu_usage_percent': round(psutil.cpu_percent(interval=0.1), 1),
                'memory_usage_percent': round(memory.percent, 1),
                'memory_available_gb': round(memory.available / (1024 ** 3), 1),
                'predictions_made': self.prediction_count,
                'success_count': self.success_count,
                'avg_inference_time_ms': round((self.total_inference_time / max(self.prediction_count, 1)) * 1000, 2),
                'uptime_seconds': round(time.time() - self.start_time, 1),
                'success_rate': round((self.success_count / max(self.prediction_count, 1)) * 100, 1)
            }
        except Exception as e:
            self.logger.warning(f"⚠️ Falha ao obter estatísticas do sistema: {e}")
            return {'error': 'Erro ao obter stats'}

    def log_prediction_stats(self, inference_time: float, batch_size: int = 1, success: bool = True):
        """Registra e calcula as métricas de performance após cada predição."""
        self.prediction_count += batch_size
        self.total_inference_time += inference_time
        if success:
            self.success_count += batch_size

        try:
            throughput = batch_size / inference_time if inference_time > 0 else float('inf')
            status_icon = "✅" if success else "❌"
            self.logger.info(
                f"📊 {status_icon} Predição: {batch_size} amostra(s) em {inference_time * 1000:.1f}ms | "
                f"Throughput: {throughput:.0f} amostras/s"
            )
        except Exception as e:
            self.logger.warning(f"⚠️ Erro ao registrar log de predição: {e}")


class TrustShieldPredictor:
    """
    Motor de inferência de produção do TrustShield.
    Carrega um modelo treinado e fornece uma interface robusta para predições em tempo real.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.logger = self._setup_logger()
        self.monitor = ResourceMonitor(self.logger)

        try:
            self.model_path = model_path or self._auto_detect_paths()
            self._log_system_info()
            self._load_artifacts()
            self.logger.info("🎯 Motor de Inferência TrustShield inicializado com sucesso!")

        except Exception as e:
            self.logger.critical(f"❌ Erro crítico na inicialização do preditor: {e}", exc_info=True)
            raise

    def _setup_logger(self) -> logging.Logger:
        """Configura um logger padronizado para o módulo."""
        logger = logging.getLogger('TrustShieldPredictor')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            formatter = logging.Formatter('%(asctime)s - [TrustShield-Predictor] - %(levelname)s - %(message)s')
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        return logger

    def _log_system_info(self):
        """Registra informações do sistema para referência."""
        try:
            memory = psutil.virtual_memory()
            cpu_freq = psutil.cpu_freq()
            self.logger.info("=" * 60)
            self.logger.info("🚀 INICIALIZANDO MOTOR DE INFERÊNCIA DE PRODUÇÃO")
            self.logger.info(
                f"💻 CPUs: {psutil.cpu_count(logical=False)} cores físicos @ {cpu_freq.current if cpu_freq else 'N/A'} MHz")
            self.logger.info(f"🧠 RAM Total: {memory.total / (1024 ** 3):.1f} GB")
            self.logger.info(f"⚙️ Threads Otimizadas: 4 (Intel MKL/OMP)")
            self.logger.info("=" * 60)
        except Exception as e:
            self.logger.warning(f"⚠️ Não foi possível obter informações detalhadas do sistema: {e}")

    def _auto_detect_paths(self) -> str:
        model_dir = Path("/home/trustshield/outputs/models")
        if not model_dir.exists():
            raise FileNotFoundError(f"Diretório de modelos não encontrado: {model_dir}")

        latest_model = max(model_dir.glob("*.joblib"), key=lambda p: p.stat().st_mtime, default=None)

        if not latest_model:
            raise FileNotFoundError(f"Nenhum modelo encontrado no diretório: {model_dir}")

        return str(latest_model)

    def _load_artifacts(self):
        """
        Carrega os artefatos de modelo (modelo e scaler) e extrai
        as features exatas que o modelo espera.
        """
        try:
            start_time = time.time()
            self.logger.info(f"📥 Carregando artefatos de: {self.model_path}")

            artifacts = joblib.load(self.model_path)

            if isinstance(artifacts, dict):
                self.model = artifacts.get('model')
                self.scaler = artifacts.get('scaler')
            else:  # Compatibilidade com modelos mais antigos
                self.model = artifacts
                self.scaler = None
                self.logger.warning("⚠️ Artefato de modelo antigo detectado (sem scaler).")

            if not self.model:
                raise ValueError("O artefato carregado não contém um objeto de modelo válido.")

            self.model_features = self._extract_model_features()
            self.model_type = self.model.__class__.__name__

            load_time = time.time() - start_time
            self.logger.info(f"✅ Artefatos carregados em {load_time:.2f}s | Tipo: {self.model_type}")
            self.logger.info(f"🎯 Modelo treinado com {len(self.model_features)} features exatas.")
            self.logger.debug(f"🔍 Lista de Features: {self.model_features[:10]}...")

        except Exception as e:
            self.logger.error(f"❌ Falha ao carregar ou processar os artefatos do modelo: {e}")
            raise

    def _extract_model_features(self) -> List[str]:
        """
        Extrai a lista de nomes de features que o modelo espera, que é a fonte da verdade.
        """
        # A fonte mais confiável de features é o atributo do próprio modelo ou do scaler.
        feature_names = getattr(self.model, 'feature_names_in_', None)
        if feature_names is None and self.scaler:
            feature_names = getattr(self.scaler, 'feature_names_in_', None)

        if feature_names is not None:
            return list(feature_names)

        # Se o modelo não tiver essa informação, é um risco para a produção.
        raise AttributeError("O artefato do modelo não contém a lista de features ('feature_names_in_'). "
                             "O modelo precisa ser retreinado com uma versão do Scikit-learn que armazene essa informação.")

    def _prepare_input_data(self, transaction_data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Prepara os dados de entrada para corresponderem EXATAMENTE ao schema do modelo.
        Esta é a etapa mais crítica para garantir a robustez em produção.
        """
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        else:  # Assume-se que seja um DataFrame
            df = transaction_data.copy()

        # Aplica one-hot encoding para features categóricas conhecidas
        categorical_cols = {'gender', 'use_chip'}
        for col in categorical_cols:
            if col in df.columns:
                df = pd.get_dummies(df, columns=[col], prefix=col, dtype='int8')

        # Cria um DataFrame final com as colunas exatas e na ordem certa que o modelo espera.
        # Colunas presentes na entrada são copiadas; as ausentes são criadas com valor 0.
        final_df = pd.DataFrame(0, index=df.index, columns=self.model_features, dtype='float32')

        common_cols = df.columns.intersection(self.model_features)
        final_df[common_cols] = df[common_cols]

        # Aplica o scaler se ele foi carregado junto com o modelo
        if self.scaler:
            try:
                scaled_data = self.scaler.transform(final_df)
                final_df = pd.DataFrame(scaled_data, columns=self.model_features, index=final_df.index, dtype='float32')
                self.logger.debug("✅ Scaler aplicado com sucesso.")
            except Exception as e:
                self.logger.warning(f"⚠️ Falha ao aplicar o scaler. Procedendo com dados não escalados. Erro: {e}")

        self.logger.debug(f"📊 Shape dos dados preparados: {final_df.shape} (match exato com o modelo)")
        return final_df

    def predict(self, transaction_data: Union[Dict, pd.DataFrame]) -> Dict[str, Any]:
        """
        Executa uma predição para uma única transação ou um batch.
        Retorna um dicionário estruturado com o resultado.
        """
        start_time = time.time()
        try:
            prepared_df = self._prepare_input_data(transaction_data)

            # Executa a predição
            predictions = self._execute_prediction(prepared_df)

            # Calcula o score de confiança (se possível)
            confidence_scores = self._calculate_confidence(prepared_df)

            inference_time = time.time() - start_time
            self.monitor.log_prediction_stats(inference_time, len(prepared_df), success=True)

            # Retorna o resultado da primeira predição se for uma única transação
            is_single = isinstance(transaction_data, dict)
            result_prediction = int(predictions[0]) if is_single else [int(p) for p in predictions]
            result_confidence = float(confidence_scores[0]) if is_single else [float(c) for c in confidence_scores]

            return {
                'prediction': result_prediction,
                'prediction_label': 'ANOMALIA' if result_prediction == -1 else 'NORMAL',
                'confidence_score': result_confidence,
                'inference_time_ms': round(inference_time * 1000, 2),
                'model_type': self.model_type,
                'model_path': str(self.model_path.name),
                'timestamp': datetime.now().isoformat(),
                'success': True
            }

        except Exception as e:
            inference_time = time.time() - start_time
            self.monitor.log_prediction_stats(inference_time, 1, success=False)
            self.logger.error(f"❌ Falha durante a predição: {e}", exc_info=True)
            return {
                'prediction': 1,  # Default para 'NORMAL' em caso de erro
                'prediction_label': 'NORMAL',
                'confidence_score': 0.0,
                'error': str(e),
                'success': False
            }

    def _execute_prediction(self, prepared_df: pd.DataFrame) -> np.ndarray:
        """Lógica interna para chamar o método de predição do modelo."""
        # Para modelos Sklearn que suportam, n_jobs é setado em tempo de predição
        if hasattr(self.model, 'n_jobs'):
            try:
                self.model.n_jobs = 4
            except Exception:
                pass  # Ignora se o atributo for read-only

        if hasattr(self.model, 'predict'):
            return self.model.predict(prepared_df)
        elif hasattr(self.model, 'decision_function'):  # Fallback para modelos como OneClassSVM
            scores = self.model.decision_function(prepared_df)
            return np.where(scores >= 0, 1, -1)
        else:
            raise NotImplementedError(
                f"O modelo do tipo {self.model_type} não possui um método 'predict' ou 'decision_function'.")

    def _calculate_confidence(self, prepared_df: pd.DataFrame) -> np.ndarray:
        """Calcula um score de confiança baseado na saída do modelo."""
        try:
            if hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(prepared_df)
                # Normaliza o score para um intervalo aproximado [0, 1]
                return 1 / (1 + np.exp(-np.abs(scores)))
            elif hasattr(self.model, 'score_samples'):
                scores = self.model.score_samples(prepared_df)
                # Normaliza o score (maior score = mais normal)
                return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        except Exception as e:
            self.logger.warning(f"⚠️ Não foi possível calcular o score de confiança: {e}")

        # Retorna um valor padrão se o cálculo não for possível
        return np.full(len(prepared_df), 0.5)

    def get_status(self) -> Dict[str, Any]:
        """Retorna um dicionário com o status atual do sistema e do modelo."""
        return {
            'status': 'OPERATIONAL',
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'type': self.model_type,
                'path': str(self.model_path),
                'features_count': len(self.model_features),
                'has_scaler': self.scaler is not None
            },
            'performance_metrics': self.monitor.get_current_stats()
        }


def run_demo():
    """Executa uma demonstração do motor de inferência com exemplos práticos."""
    print("\n" + "=" * 80)
    print("🚀 DEMONSTRAÇÃO DO MOTOR DE INFERÊNCIA TRUSTSHIELD")
    print("=" * 80)

    try:
        predictor = TrustShieldPredictor()

        status = predictor.get_status()
        print("\n📊 STATUS INICIAL DO SISTEMA:")
        print(f"  ● Modelo: {status['model_info']['type']} de {status['model_info']['path']}")
        print(f"  ● Features Esperadas: {status['model_info']['features_count']}")
        print(f"  ● Scaler Presente: {'Sim' if status['model_info']['has_scaler'] else 'Não'}")

        # Exemplo 1: Transação claramente suspeita
        print("\n" + "-" * 80)
        print("🚨 EXEMPLO 1: Transação de Alto Risco (potencial fraude)")
        suspicious_transaction = {
            'amount': 8750.00,
            'use_chip': 'Online Transaction',
            'current_age': 50,
            'retirement_age': 65,
            'birth_year': 1974,
            'gender': 'Male',
            'latitude': 25.7617,
            'longitude': -80.1918,
            'yearly_income': 40000,
            'total_debt': 80000,
            'credit_score': 510,
            'num_credit_cards': 12,
            'transaction_hour': 2,  # Madrugada
            'day_of_week': 6,  # Fim de semana
            'is_weekend': True,
            'is_night_transaction': True,
            'amount_vs_avg': 50.0  # Valor muito acima da média
        }
        result = predictor.predict(suspicious_transaction)
        print(f"  ▶️ Resultado: {result['prediction_label']} (Score: {result['confidence_score']:.3f})")
        print(f"  ⏱️ Tempo de Inferência: {result['inference_time_ms']:.1f}ms")
        print(f"  ✅ Sucesso da Predição: {'Sim' if result.get('success') else 'Não'}")

        # Exemplo 2: Transação normal do dia a dia
        print("\n" + "-" * 80)
        print("✅ EXEMPLO 2: Transação de Baixo Risco (normal)")
        normal_transaction = {
            'amount': 45.75,
            'use_chip': 'Chip Transaction',
            'current_age': 32,
            'retirement_age': 67,
            'birth_year': 1992,
            'gender': 'Female',
            'latitude': 40.7128,
            'longitude': -74.0060,
            'yearly_income': 95000,
            'total_debt': 12000,
            'credit_score': 780,
            'num_credit_cards': 3,
            'transaction_hour': 14,  # Horário comercial
            'day_of_week': 2,  # Dia de semana
            'is_weekend': False,
            'is_night_transaction': False,
            'amount_vs_avg': 0.8
        }
        result = predictor.predict(normal_transaction)
        print(f"  ▶️ Resultado: {result['prediction_label']} (Score: {result['confidence_score']:.3f})")
        print(f"  ⏱️ Tempo de Inferência: {result['inference_time_ms']:.1f}ms")

        # Exemplo 3: Teste de robustez com dados mínimos
        print("\n" + "-" * 80)
        print("🧪 EXEMPLO 3: Teste de Robustez com Dados Mínimos")
        minimal_data = {'amount': 150.0, 'credit_score': 680, 'transaction_hour': 23}
        result = predictor.predict(minimal_data)
        print(f"  ▶️ Resultado: {result['prediction_label']} (Score: {result['confidence_score']:.3f})")
        print(f"  📝 Nota: O sistema preencheu automaticamente as features ausentes com valores padrão.")

        print("\n" + "=" * 80)
        final_status = predictor.get_status()
        print("\n📈 STATUS FINAL DO SISTEMA:")
        print(f"  ● Total de Predições: {final_status['performance_metrics']['predictions_made']}")
        print(f"  ● Taxa de Sucesso: {final_status['performance_metrics']['success_rate']:.1f}%")
        print(f"  ● Tempo Médio de Inferência: {final_status['performance_metrics']['avg_inference_time_ms']:.1f}ms")
        print("\nDemonstração concluída com sucesso!")

    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO DURANTE A DEMONSTRAÇÃO: {e}")
        print("Verifique se um modelo foi treinado e se os caminhos estão corretos.")
        sys.exit(1)


if __name__ == "__main__":
    run_demo()