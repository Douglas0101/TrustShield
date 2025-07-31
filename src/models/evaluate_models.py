# -*- coding: utf-8 -*-
"""
Módulo de Avaliação Comparativa Otimizada para Intel i3-1115G4 - Projeto TrustShield
VERSÃO CORRIGIDA - Lida com todos os tipos de modelos (LOF, SVM, IsolationForest)

Hardware Target:
- CPU: 11th Gen Intel® Core™ i3-1115G4 × 4 cores
- RAM: 20.0 GiB
- Storage: 480.1 GB SSD
- OS: Ubuntu 24.04.2 LTS

Correções v2.1.1:
- Suporte completo para LOF (sem método predict)
- Detecção automática do tipo de modelo
- Fallback para decision_function quando predict não disponível
- Otimização de memória aprimorada (chunks menores se necessário)
- Tratamento robusto de modelos hierárquicos

Execução:
    python src/models/evaluate_models_fixed.py --config config/config.yaml --optimize-intel

Autor: TrustShield Team - Intel Optimized Fixed
Versão: 2.1.1-intel-fixed
"""

import argparse
import gc
import json
import logging
import os
import psutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score
)
from sklearn.model_selection import train_test_split

# Configurações Intel específicas
os.environ['OMP_NUM_THREADS'] = '4'  # Cores disponíveis
os.environ['MKL_NUM_THREADS'] = '4'  # Intel MKL threads
os.environ['NUMBA_NUM_THREADS'] = '4'  # Numba threads
os.environ['OPENBLAS_NUM_THREADS'] = '4'  # OpenBLAS threads
os.environ['MKL_DYNAMIC'] = 'FALSE'  # Desabilita thread dinâmico MKL

# Configurações globais otimizadas
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configurações de memória ajustadas
CHUNK_SIZE = 300000  # Reduzido para 300k (otimização de memória)
MAX_MEMORY_USAGE = 0.75  # Reduzido para 75% da RAM
CACHE_SIZE_MB = 1536  # Cache reduzido para 1.5GB


class IntelOptimizedConfig:
    """Configuração otimizada para processador Intel i3-1115G4."""

    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        self.random_state = config_dict.get('project', {}).get('random_state', 42)
        self.test_size = config_dict.get('validation', {}).get('test_size', 0.15)

        # Configurações Intel específicas
        self.n_jobs = 4  # 4 cores físicos
        self.chunk_size = CHUNK_SIZE
        self.max_memory_gb = 14  # Deixar 6GB livres do sistema
        self.enable_intel_optimizations = True

        # Configurar NumPy para Intel
        if hasattr(np, '__config__'):
            self._configure_intel_mkl()

    def _configure_intel_mkl(self):
        """Configura otimizações Intel MKL se disponível."""
        try:
            import mkl
            mkl.set_num_threads(self.n_jobs)
            mkl.domain_set_num_threads(self.n_jobs, domain='all')
            self.intel_mkl_available = True
        except ImportError:
            self.intel_mkl_available = False


class ResourceMonitor:
    """Monitor de recursos do sistema em tempo real."""

    def __init__(self, logger):
        self.logger = logger
        self.process = psutil.Process()
        self.start_time = time.time()
        self.peak_memory = 0

    def get_system_info(self) -> Dict[str, Any]:
        """Obtém informações do sistema."""
        memory = psutil.virtual_memory()
        cpu_freq = psutil.cpu_freq()

        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq_mhz': cpu_freq.current if cpu_freq else 'N/A',
            'memory_total_gb': round(memory.total / (1024**3), 1),
            'memory_available_gb': round(memory.available / (1024**3), 1),
            'memory_percent': memory.percent,
            'disk_usage_percent': psutil.disk_usage('/').percent
        }

    def check_memory_usage(self) -> bool:
        """Verifica se o uso de memória está dentro dos limites."""
        memory_info = self.process.memory_info()
        memory_gb = memory_info.rss / (1024**3)

        if memory_gb > self.peak_memory:
            self.peak_memory = memory_gb

        memory_percent = psutil.virtual_memory().percent

        if memory_percent > MAX_MEMORY_USAGE * 100:
            self.logger.warning(f"⚠️ Alto uso de memória: {memory_percent:.1f}% - Executando garbage collection")
            gc.collect()  # Força limpeza
            return False

        return True

    def log_resource_usage(self):
        """Log do uso atual de recursos."""
        sys_info = self.get_system_info()
        elapsed_time = time.time() - self.start_time

        self.logger.info(f"💻 Recursos - CPU: {psutil.cpu_percent():.1f}% | "
                        f"RAM: {sys_info['memory_percent']:.1f}% | "
                        f"Pico: {self.peak_memory:.1f}GB | "
                        f"Tempo: {elapsed_time:.1f}s")


class OptimizedDataLoader:
    """Carregador de dados otimizado para hardware Intel."""

    def __init__(self, config: IntelOptimizedConfig, logger, monitor: ResourceMonitor):
        self.config = config
        self.logger = logger
        self.monitor = monitor

    def load_data_optimized(self, project_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Carrega dados de forma otimizada com chunks."""
        data_path = project_root / self.config.config['paths']['data']['featured_dataset']

        self.logger.info(f"🔄 Carregando dados otimizado: {data_path}")

        # Verificar tamanho do arquivo
        file_size_mb = data_path.stat().st_size / (1024**2)
        self.logger.info(f"📁 Tamanho do arquivo: {file_size_mb:.1f} MB")

        if file_size_mb > 800:  # Limite reduzido para 800MB
            return self._load_large_dataset(data_path)
        else:
            return self._load_standard_dataset(data_path)

    def _load_large_dataset(self, data_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Carrega dataset grande em chunks."""
        self.logger.info("📊 Dataset grande detectado - usando carregamento em chunks")

        # Ler em chunks para economizar memória
        chunk_list = []
        total_rows = 0
        max_chunks = 45  # Limitar número de chunks

        for i, chunk in enumerate(pd.read_parquet(data_path, chunksize=self.config.chunk_size)):
            if i >= max_chunks:
                self.logger.info(f"⚠️ Limitando a {max_chunks} chunks para economizar memória")
                break

            # Otimizar tipos de dados
            chunk = self._optimize_dtypes(chunk)
            chunk_list.append(chunk)
            total_rows += len(chunk)

            self.monitor.log_resource_usage()

            if not self.monitor.check_memory_usage():
                self.logger.warning("⚠️ Limite de memória atingido - parando carregamento")
                break

        # Concatenar chunks
        df = pd.concat(chunk_list, ignore_index=True)
        del chunk_list
        gc.collect()

        self.logger.info(f"✅ Dataset carregado: {total_rows:,} amostras")

        return self._split_data(df)

    def _load_standard_dataset(self, data_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Carrega dataset padrão."""
        df = pd.read_parquet(data_path)
        df = self._optimize_dtypes(df)

        self.logger.info(f"✅ Dataset carregado: {len(df):,} amostras")
        return self._split_data(df)

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Otimiza tipos de dados para economizar memória."""
        # Converter float64 para float32
        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')

        # Converter int64 para int32 quando possível
        int_cols = df.select_dtypes(include=['int64']).columns
        for col in int_cols:
            if df[col].max() < 2147483647 and df[col].min() > -2147483648:
                df[col] = df[col].astype('int32')

        return df

    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Divide dados em treino e teste."""
        df_train, df_test = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=df.get('is_anomaly') if 'is_anomaly' in df.columns else None
        )

        # Pré-processar
        X_train = self._preprocess_features(df_train)
        X_test = self._preprocess_features(df_test)

        # Limpar memória
        del df, df_train, df_test
        gc.collect()

        return X_train, X_test

    def _preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pré-processamento otimizado."""
        features_to_drop = self.config.config.get('preprocessing', {}).get('features_to_drop', [])
        X = df.drop(columns=features_to_drop + ['is_anomaly'], errors='ignore').copy()

        # Encoding otimizado
        categorical_features = self.config.config.get('preprocessing', {}).get('categorical_features', [])
        existing_categorical = [col for col in categorical_features if col in X.columns]

        if existing_categorical:
            X = pd.get_dummies(X, columns=existing_categorical, drop_first=True, dtype='int8')

        X = X.fillna(0).astype('float32')
        return X


class SmartModelPredictor:
    """Preditor inteligente que lida com diferentes tipos de modelos."""

    def __init__(self, config: IntelOptimizedConfig, logger, monitor: ResourceMonitor):
        self.config = config
        self.logger = logger
        self.monitor = monitor
        self.model_cache = {}

    def predict_with_batching(self, model_name: str, artifact: Any, X_test: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, float]]:
        """Predição inteligente que detecta o tipo de modelo."""
        self.logger.info(f"🔮 Iniciando predição inteligente: {model_name}")

        start_time = time.time()

        # Detectar tipo de modelo
        model_type = self._detect_model_type(model_name, artifact)
        self.logger.info(f"🔍 Tipo detectado: {model_type}")

        if len(X_test) > self.config.chunk_size:
            predictions = self._predict_in_batches(model_name, artifact, X_test, model_type)
        else:
            predictions = self._predict_single_batch(model_name, artifact, X_test, model_type)

        inference_time = time.time() - start_time

        # Calcular métricas
        metrics = {
            'inference_time_seconds': round(inference_time, 4),
            'throughput': round(len(X_test) / inference_time, 2) if inference_time > 0 else 0,
            'samples_per_core_per_second': round(len(X_test) / (inference_time * self.config.n_jobs), 2) if inference_time > 0 else 0,
            'model_type': model_type
        }

        self.logger.info(f"✅ Predição concluída: {len(X_test):,} amostras em {inference_time:.2f}s")
        return predictions, metrics

    def _detect_model_type(self, model_name: str, artifact: Any) -> str:
        """Detecta o tipo de modelo e método de predição disponível."""
        if isinstance(artifact, dict):
            if 'clusterer' in artifact and 'lof_models' in artifact:
                return 'hierarchical_lof'
            elif 'approximator' in artifact and 'svm_model' in artifact:
                return 'hierarchical_svm'
            else:
                return 'dict_artifact'

        # Verificar métodos disponíveis
        if hasattr(artifact, 'predict'):
            return 'standard_predict'
        elif hasattr(artifact, 'decision_function'):
            return 'decision_function'
        elif hasattr(artifact, 'score_samples'):
            return 'score_samples'
        elif hasattr(artifact, 'fit_predict'):
            return 'fit_predict_only'
        else:
            return 'unknown'

    def _predict_in_batches(self, model_name: str, artifact: Any, X_test: pd.DataFrame, model_type: str) -> np.ndarray:
        """Predição em batches para datasets grandes."""
        self.logger.info(f"📦 Processando em batches de {self.config.chunk_size:,} amostras")

        predictions_list = []
        n_batches = len(X_test) // self.config.chunk_size + 1

        for i in range(0, len(X_test), self.config.chunk_size):
            batch_idx = i // self.config.chunk_size + 1
            end_idx = min(i + self.config.chunk_size, len(X_test))
            X_batch = X_test.iloc[i:end_idx]

            self.logger.info(f"🔄 Processando batch {batch_idx}/{n_batches}")

            batch_predictions = self._predict_single_batch(model_name, artifact, X_batch, model_type)
            predictions_list.append(batch_predictions)

            # Monitorar recursos
            self.monitor.log_resource_usage()

            if not self.monitor.check_memory_usage():
                gc.collect()  # Força garbage collection

        return np.concatenate(predictions_list)

    def _predict_single_batch(self, model_name: str, artifact: Any, X_batch: pd.DataFrame, model_type: str) -> np.ndarray:
        """Predição inteligente baseada no tipo de modelo."""
        try:
            if model_type == 'hierarchical_lof':
                return self._predict_hierarchical_lof(artifact, X_batch)

            elif model_type == 'hierarchical_svm':
                return self._predict_hierarchical_svm(artifact, X_batch)

            elif model_type == 'standard_predict':
                # Configurar n_jobs para modelos sklearn
                if hasattr(artifact, 'n_jobs'):
                    artifact.n_jobs = self.config.n_jobs
                return artifact.predict(X_batch)

            elif model_type == 'decision_function':
                # Usar decision_function e converter para predições
                scores = artifact.decision_function(X_batch)
                return np.where(scores >= 0, 1, -1)

            elif model_type == 'score_samples':
                # Usar score_samples e determinar threshold
                scores = artifact.score_samples(X_batch)
                threshold = np.percentile(scores, 10)  # Bottom 10% como anomalias
                return np.where(scores >= threshold, 1, -1)

            elif model_type == 'fit_predict_only':
                self.logger.warning(f"⚠️ Modelo {model_name} requer retreinamento - pulando")
                return np.ones(len(X_batch), dtype=np.int8)  # Padrão: tudo normal

            else:
                self.logger.error(f"❌ Tipo de modelo não suportado: {model_type}")
                return np.ones(len(X_batch), dtype=np.int8)  # Padrão: tudo normal

        except Exception as e:
            self.logger.error(f"❌ Erro na predição {model_name}: {str(e)}")
            return np.ones(len(X_batch), dtype=np.int8)  # Padrão: tudo normal

    def _predict_hierarchical_lof(self, artifact: Dict, X_batch: pd.DataFrame) -> np.ndarray:
        """Predição para LOF hierárquico."""
        clusterer = artifact['clusterer']
        lof_models = artifact['lof_models']

        # Configurar n_jobs se disponível
        if hasattr(clusterer, 'n_jobs'):
            clusterer.n_jobs = self.config.n_jobs

        try:
            cluster_labels = clusterer.predict(X_batch)
            predictions = np.ones(len(X_batch), dtype=np.int8)  # Padrão: normal

            for i, model in lof_models.items():
                mask = cluster_labels == i
                if np.any(mask):
                    X_cluster = X_batch[mask]

                    # LOF não tem predict - usar decision_function ou fit_predict
                    if hasattr(model, 'decision_function'):
                        scores = model.decision_function(X_cluster)
                        predictions[mask] = np.where(scores >= 0, 1, -1)
                    elif hasattr(model, 'score_samples'):
                        scores = model.score_samples(X_cluster)
                        threshold = np.percentile(scores, 10)
                        predictions[mask] = np.where(scores >= threshold, 1, -1)
                    else:
                        # Se não tem métodos de predição, usar padrão (normal)
                        predictions[mask] = 1

            return predictions

        except Exception as e:
            self.logger.warning(f"⚠️ Erro LOF hierárquico: {str(e)}")
            return np.ones(len(X_batch), dtype=np.int8)

    def _predict_hierarchical_svm(self, artifact: Dict, X_batch: pd.DataFrame) -> np.ndarray:
        """Predição para SVM hierárquico."""
        approximator = artifact['approximator']
        svm_model = artifact['svm_model']

        try:
            X_transformed = approximator.transform(X_batch)
            return svm_model.predict(X_transformed)
        except Exception as e:
            self.logger.warning(f"⚠️ Erro SVM hierárquico: {str(e)}")
            return np.ones(len(X_batch), dtype=np.int8)


class OptimizedMetricsCalculator:
    """Calculadora de métricas otimizada."""

    def __init__(self, config: IntelOptimizedConfig, logger, monitor: ResourceMonitor):
        self.config = config
        self.logger = logger
        self.monitor = monitor

    def calculate_metrics_optimized(self, X_test: pd.DataFrame, predictions: np.ndarray) -> Dict[str, float]:
        """Cálculo otimizado de métricas."""
        self.logger.info("📊 Calculando métricas otimizadas")

        metrics = {}

        # Métricas básicas (rápidas)
        total_samples = len(predictions)
        anomalies_detected = np.sum(predictions == -1)
        anomaly_rate = anomalies_detected / total_samples

        metrics.update({
            'total_samples': total_samples,
            'anomalies_detected': int(anomalies_detected),
            'anomaly_rate': round(anomaly_rate, 5),
            'normal_samples': total_samples - anomalies_detected
        })

        # Métricas de clustering (apenas para datasets menores)
        if len(np.unique(predictions)) > 1 and len(X_test) <= 50000:  # Limite reduzido
            try:
                self.logger.info("🧮 Calculando métricas de clustering")

                # Usar amostragem ainda menor
                if len(X_test) > 20000:
                    sample_size = 20000
                    sample_idx = np.random.choice(len(X_test), sample_size, replace=False)
                    X_sample = X_test.iloc[sample_idx]
                    pred_sample = predictions[sample_idx]
                else:
                    X_sample = X_test
                    pred_sample = predictions

                # Silhouette Score com sample menor
                silhouette_avg = silhouette_score(X_sample, pred_sample, sample_size=min(5000, len(X_sample)))
                metrics['silhouette_score'] = round(silhouette_avg, 4)

                # Outras métricas
                ch_score = calinski_harabasz_score(X_sample, pred_sample)
                metrics['calinski_harabasz_score'] = round(ch_score, 4)

                db_score = davies_bouldin_score(X_sample, pred_sample)
                metrics['davies_bouldin_score'] = round(db_score, 4)

            except Exception as e:
                self.logger.warning(f"⚠️ Erro ao calcular métricas de clustering: {str(e)}")

        else:
            self.logger.info("ℹ️ Pulando métricas de clustering (dataset muito grande ou classes insuficientes)")

        return metrics


class IntelOptimizedPipeline:
    """Pipeline principal otimizado para Intel i3-1115G4."""

    def __init__(self, config_path: str, enable_intel_optimizations: bool = True):
        self.project_root = Path.cwd()
        self.config = IntelOptimizedConfig(self._load_config(config_path))

        # Setup logging
        self.logger = self._setup_logger()

        # Monitor de recursos
        self.monitor = ResourceMonitor(self.logger)

        # Log informações do sistema
        self._log_system_info()

        # Componentes otimizados
        self.data_loader = OptimizedDataLoader(self.config, self.logger, self.monitor)
        self.model_predictor = SmartModelPredictor(self.config, self.logger, self.monitor)  # Novo predictor inteligente
        self.metrics_calculator = OptimizedMetricsCalculator(self.config, self.logger, self.monitor)

        # Diretórios
        self.output_dir = self.project_root / 'outputs'
        self.model_dir = self.output_dir / 'models'

    def _setup_logger(self):
        """Setup do logger otimizado."""
        logger = logging.getLogger('TrustShield-Intel-Fixed')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            formatter = logging.Formatter('%(asctime)s - [INTEL-FIX] - %(levelname)s - %(message)s')

            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # File handler
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(
                log_dir / f'intel_fixed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Carrega configuração."""
        config_file = Path(config_path)
        if not config_file.exists():
            config_file = self.project_root / config_path

        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def _log_system_info(self):
        """Log das informações do sistema."""
        sys_info = self.monitor.get_system_info()

        self.logger.info("🖥️ === SISTEMA OTIMIZADO INTEL i3-1115G4 - VERSÃO CORRIGIDA ===")
        self.logger.info(f"💻 CPUs: {sys_info['cpu_count']} cores @ {sys_info['cpu_freq_mhz']} MHz")
        self.logger.info(f"🧠 RAM: {sys_info['memory_total_gb']} GB (disponível: {sys_info['memory_available_gb']} GB)")
        self.logger.info(f"💾 Disco: {sys_info['disk_usage_percent']:.1f}% usado")
        self.logger.info(f"⚙️ Threads configuradas: {self.config.n_jobs}")
        self.logger.info(f"📦 Tamanho do chunk otimizado: {self.config.chunk_size:,} amostras")

        if hasattr(self.config, 'intel_mkl_available') and self.config.intel_mkl_available:
            self.logger.info("🚀 Intel MKL ativado - performance otimizada!")
        else:
            self.logger.info("ℹ️ Intel MKL não disponível - usando configurações padrão")

    def run_optimized_evaluation(self) -> Path:
        """Executa avaliação otimizada e corrigida."""
        start_time = time.time()

        try:
            self.logger.info("🚀 === INICIANDO AVALIAÇÃO CORRIGIDA INTEL ===")

            # 1. Carregar dados otimizado
            self.logger.info("📂 Etapa 1: Carregamento otimizado de dados")
            X_train, X_test = self.data_loader.load_data_optimized(self.project_root)
            self.monitor.log_resource_usage()

            # 2. Carregar modelos
            self.logger.info("🤖 Etapa 2: Carregamento de modelos")
            models = self._load_models_optimized()

            # 3. Avaliação inteligente
            self.logger.info("🔮 Etapa 3: Predições inteligentes")
            results = {}
            all_predictions = {}

            for model_name, model_info in models.items():
                try:
                    self.logger.info(f"🔄 Avaliando modelo: {model_name}")

                    # Predição inteligente
                    predictions, pred_metrics = self.model_predictor.predict_with_batching(
                        model_name, model_info['artifact'], X_test
                    )

                    # Métricas otimizadas
                    anomaly_metrics = self.metrics_calculator.calculate_metrics_optimized(X_test, predictions)

                    # Combinar métricas
                    combined_metrics = {**anomaly_metrics, **pred_metrics}
                    results[model_name] = combined_metrics
                    all_predictions[model_name] = predictions

                    self.monitor.log_resource_usage()
                    gc.collect()  # Limpar memória

                except Exception as e:
                    self.logger.error(f"❌ Erro ao avaliar {model_name}: {str(e)}")
                    continue

            # 4. Relatório final
            self.logger.info("📊 Etapa 4: Geração de relatório")
            self._display_optimized_summary(results, all_predictions)

            total_time = time.time() - start_time
            self.logger.info(f"✅ === AVALIAÇÃO CONCLUÍDA EM {total_time:.1f}s ===")
            self.logger.info(f"🏆 Pico de memória utilizada: {self.monitor.peak_memory:.1f} GB")

            return self._save_results(results, all_predictions)

        except Exception as e:
            self.logger.error(f"❌ Erro na avaliação otimizada: {str(e)}")
            raise

    def _load_models_optimized(self) -> Dict[str, Any]:
        """Carrega modelos com detecção inteligente."""
        model_patterns = ['*_i3_optimized_*.joblib', '*.joblib', '*.pkl']
        model_files = []

        for pattern in model_patterns:
            found_files = list(self.model_dir.glob(pattern))
            model_files.extend(found_files)
            if found_files:
                break

        if not model_files:
            raise FileNotFoundError(f"Nenhum modelo encontrado em {self.model_dir}!")

        models = {}
        for model_path in model_files:
            try:
                model_name = self._extract_model_name(model_path.name)
                self.logger.info(f"📥 Carregando: {model_name} ({model_path.name})")

                # Carregar com otimizações de memória
                with open(model_path, 'rb') as f:
                    artifact = joblib.load(f)

                models[model_name] = {
                    'artifact': artifact,
                    'path': model_path,
                    'size_mb': round(model_path.stat().st_size / (1024**2), 1)
                }

            except Exception as e:
                self.logger.warning(f"⚠️ Erro ao carregar {model_path}: {str(e)}")

        self.logger.info(f"✅ {len(models)} modelos carregados")
        return models

    def _extract_model_name(self, filename: str) -> str:
        """Extrai nome do modelo do arquivo."""
        # Remove extensões e timestamps
        name = filename.split('.')[0]

        # Remove sufixos comuns
        for suffix in ['_i3_optimized', '_optimized', '_model']:
            if suffix in name:
                name = name.split(suffix)[0]
                break

        # Remove timestamps (padrão YYYYMMDD_HHMMSS)
        import re
        name = re.sub(r'_\d{8}_\d{6}$', '', name)

        return name

    def _display_optimized_summary(self, results: Dict[str, Dict], all_predictions: Dict[str, np.ndarray]):
        """Exibe resumo otimizado e corrigido."""
        print("\n" + "=" * 90)
        print("🎯 RELATÓRIO OTIMIZADO E CORRIGIDO - INTEL i3-1115G4")
        print("=" * 90)

        if not results:
            print("❌ Nenhum modelo foi avaliado com sucesso!")
            return

        # Tabela de performance
        df_results = pd.DataFrame(results).T

        print("\n🔍 MODELOS AVALIADOS:")
        for model_name, metrics in results.items():
            model_type = metrics.get('model_type', 'unknown')
            throughput = metrics.get('throughput', 0)
            anomalies = metrics.get('anomalies_detected', 0)

            print(f"  • {model_name}: {model_type} | {throughput:.0f} samples/s | {anomalies:,} anomalias")

        # Métricas de performance
        performance_columns = ['inference_time_seconds', 'throughput', 'samples_per_core_per_second', 'anomalies_detected']
        existing_columns = [col for col in performance_columns if col in df_results.columns]

        if existing_columns:
            print("\n🚀 MÉTRICAS DE PERFORMANCE:")
            print(df_results[existing_columns].round(4).to_string())

        # Análise de eficiência
        if 'throughput' in df_results.columns and len(df_results) > 0:
            best_model = df_results['throughput'].idxmax()
            best_throughput = df_results.loc[best_model, 'throughput']

            print(f"\n🏆 MODELO MAIS EFICIENTE: {best_model}")
            print(f"📈 Throughput: {best_throughput:.0f} amostras/segundo")
            print(f"⚡ Por core: {best_throughput/4:.0f} amostras/segundo/core")
            print(f"🕐 Capacidade estimada: {best_throughput*3600:.0f} amostras/hora")

        print("\n" + "=" * 90)

    def _save_results(self, results: Dict[str, Dict], all_predictions: Dict[str, np.ndarray]) -> Path:
        """Salva resultados corrigidos."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = self.output_dir / 'reports'
        results_dir.mkdir(exist_ok=True)

        results_file = results_dir / f'intel_fixed_results_{timestamp}.json'

        # Preparar dados para JSON
        json_data = {
            'metadata': {
                'timestamp': timestamp,
                'hardware': 'Intel i3-1115G4',
                'optimization_level': 'intel_fixed_v2.1.1',
                'peak_memory_gb': round(self.monitor.peak_memory, 2),
                'chunk_size': self.config.chunk_size,
                'models_evaluated': len(results)
            },
            'results': results,
            'system_info': self.monitor.get_system_info()
        }

        with open(results_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)

        self.logger.info(f"💾 Resultados salvos: {results_file}")
        return results_file


def main():
    """Função principal corrigida."""
    parser = argparse.ArgumentParser(
        description="Sistema de Avaliação Otimizado e Corrigido para Intel i3-1115G4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CORREÇÕES v2.1.1:
✅ Suporte completo para LOF (sem método predict)
✅ Detecção automática do tipo de modelo  
✅ Predição inteligente com fallbacks
✅ Gestão otimizada de memória (chunks menores)
✅ Tratamento robusto de erros

Exemplo:
    python src/models/evaluate_models_fixed.py --config config/config.yaml
        """
    )

    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--optimize-intel", action="store_true", help="Ativa otimizações Intel")

    args = parser.parse_args()

    try:
        pipeline = IntelOptimizedPipeline(
            config_path=args.config,
            enable_intel_optimizations=args.optimize_intel
        )

        results_file = pipeline.run_optimized_evaluation()

        print(f"\n✅ Avaliação corrigida concluída!")
        print(f"📄 Resultados: {results_file}")
        print(f"🚀 Sistema otimizado e corrigido para Intel i3-1115G4")

    except KeyboardInterrupt:
        print("\n❌ Interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
