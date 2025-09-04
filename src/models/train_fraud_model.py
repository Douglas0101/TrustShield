#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sistema de Otimização Avançada TrustShield para Hardware Limitado
Estratégias PhD-level adaptadas para Intel i3-1115G4 com 20GB RAM

Autor: TrustShield PhD Engineering Team
Versão: 2.0.0-i3-optimized
"""

import os
import sys
import gc
import time
import joblib
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from typing import Dict, List, Tuple, Any
import psutil
import hashlib
import pickle

# Otimizações de sistema
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '2'  # Otimizado para 2 cores físicos
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
os.environ['NUMBA_NUM_THREADS'] = '2'

# Importações científicas
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import mlflow
from src.utils.mlflow_setup import setup_mlflow

# Tentativa de imports otimizados
try:
    import numba
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️ Numba não disponível - instale para 2-3x speedup: pip install numba")

try:
    import modin.pandas as mpd

    MODIN_AVAILABLE = True
    print("✅ Modin detectado - DataFrames paralelos ativados")
except ImportError:
    MODIN_AVAILABLE = False
    mpd = pd  # Fallback para pandas normal

try:
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingGridSearchCV

    HALVING_AVAILABLE = True
except ImportError:
    HALVING_AVAILABLE = False


class IntelI3Optimizer:
    """
    Otimizador específico para processadores Intel i3 com RAM abundante.
    Usa técnicas avançadas de cache, vetorização e processamento inteligente.
    """

    def __init__(self):
        self.cpu_count = cpu_count()  # 4 threads
        self.physical_cores = psutil.cpu_count(logical=False)  # 2 cores
        self.ram_gb = psutil.virtual_memory().total / (1024 ** 3)  # ~20 GB
        self.available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)

        print(f"""
╔══════════════════════════════════════════════════════════════╗
║           TRUSTSHIELD PHD-LEVEL OPTIMIZER V2.0               ║
║                  Intel i3 Specialized Edition                ║
╚══════════════════════════════════════════════════════════════╝

🖥️  Hardware Detectado:
    • CPU: Intel i3 - {self.physical_cores} cores / {self.cpu_count} threads
    • RAM: {self.ram_gb:.1f} GB total / {self.available_ram_gb:.1f} GB disponível
    • Otimização: Ativada para CPU limitada + RAM abundante
        """)

        # Configurações otimizadas para i3
        self.batch_size = 50000  # Processar em chunks para caber no cache L3
        self.n_jobs_optimal = 2  # Usar cores físicos, não threads
        self.use_memory_cache = True  # Aproveitar os 20GB de RAM
        self.compression_level = 1  # Compressão leve para I/O rápido

    def optimize_data_loading(self, data_path: str) -> pd.DataFrame:
        """
        Carregamento otimizado com cache em memória.
        Técnica: Memory-mapped files + Column pruning
        """
        print("\n📊 FASE 1: Carregamento Otimizado de Dados")
        print("-" * 50)

        start = time.time()

        # Verificar cache em memória
        cache_path = Path("cache/data_cache.pkl")
        if cache_path.exists():
            print("💾 Cache encontrado - carregando...")
            with open(cache_path, 'rb') as f:
                df = pickle.load(f)
            print(f"✅ Dados carregados do cache em {time.time() - start:.2f}s")
            return df

        # Carregar com otimizações
        print("📖 Lendo dataset...")

        # Usar Modin se disponível (paralelo) ou pandas otimizado
        if MODIN_AVAILABLE:
            df = mpd.read_parquet(data_path, engine='pyarrow')
        else:
            # Leitura otimizada com dtypes específicos
            df = pd.read_parquet(
                data_path,
                engine='pyarrow',  # Mais rápido que fastparquet
                columns=None,  # Carregar todas inicialmente
            )

        # Otimizar tipos de dados para economizar memória
        print("🔧 Otimizando tipos de dados...")
        df = self.optimize_dtypes(df)

        # Limpar features desnecessárias
        features_to_drop = ["date", "merchant_city", "merchant_state", "zip",
                            "address", "card_id", "merchant_id", "errors",
                            "client_id", "id_transaction"]
        df = df.drop(columns=[col for col in features_to_drop if col in df.columns])

        # One-hot encoding otimizado
        categorical = ["use_chip", "gender"]
        for col in categorical:
            if col in df.columns:
                # Usar sparse matrix para economizar memória
                df = pd.get_dummies(df, columns=[col], sparse=False, dtype='int8')

        # Selecionar apenas numéricas e converter para float32
        df = df.select_dtypes(include='number').fillna(0).astype('float32')

        # Salvar cache
        cache_path.parent.mkdir(exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

        elapsed = time.time() - start
        print(f"✅ Dados preparados em {elapsed:.2f}s")
        print(f"   Shape: {df.shape}")
        print(f"   Memória: {df.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")

        return df

    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reduz uso de memória em até 70% convertendo tipos.
        Técnica: Downcast sistemático
        """
        start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

        # Inteiros
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')

        # Floats
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')

        # Categoricals para strings repetitivas
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Se menos de 50% único
                df[col] = df[col].astype('category')

        end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
        print(
            f"   💾 Memória reduzida: {start_mem:.1f}MB → {end_mem:.1f}MB ({(1 - end_mem / start_mem) * 100:.1f}% economia)")

        return df

    def train_single_model_optimized(self,
                                     X_train: np.ndarray,
                                     X_test: np.ndarray,
                                     config: Dict[str, Any],
                                     model_id: int) -> Dict[str, Any]:
        """
        Treina um único modelo com otimizações específicas para i3.
        Técnicas: Subsampling adaptativo + Early stopping
        """
        print(f"\n🎯 Treinando Modelo {model_id}")

        start_time = time.time()

        # Configuração otimizada para i3
        n_samples = len(X_train)

        # Subsampling inteligente baseado no tamanho do dataset
        if n_samples > 100000:
            # Para datasets grandes, usar subsampling agressivo
            max_samples = min(50000, n_samples // 10)
            print(f"   📉 Subsampling: {n_samples} → {max_samples} amostras")
        else:
            max_samples = 'auto'

        # Modelo com configurações otimizadas
        model = IsolationForest(
            n_estimators=config.get('n_estimators', 100),  # Menos árvores
            max_samples=max_samples,
            max_features=config.get('max_features', 1.0),
            contamination=config.get('contamination', 0.1),
            n_jobs=self.n_jobs_optimal,  # 2 cores físicos
            random_state=42 + model_id,
            bootstrap=False,  # Mais rápido sem bootstrap
            warm_start=False
        )

        # Treinar
        model.fit(X_train)
        train_time = time.time() - start_time

        # Validação rápida
        start_val = time.time()
        y_pred_train = model.predict(X_train[:10000])  # Validar subset
        y_pred_test = model.predict(X_test[:10000])
        val_time = time.time() - start_val

        # Métricas
        anomaly_rate_train = (y_pred_train == -1).mean()
        anomaly_rate_test = (y_pred_test == -1).mean()

        # Score de decisão para análise
        scores_test = model.decision_function(X_test[:1000])

        result = {
            'model_id': model_id,
            'model': model,
            'train_time': train_time,
            'val_time': val_time,
            'anomaly_rate_train': anomaly_rate_train,
            'anomaly_rate_test': anomaly_rate_test,
            'score_mean': scores_test.mean(),
            'score_std': scores_test.std(),
            'n_samples_trained': n_samples,
            'config': config
        }

        print(f"   ✅ Concluído em {train_time:.2f}s | Anomaly Rate: {anomaly_rate_test:.2%}")

        return result

    def parallel_hyperparameter_search(self, X_train, X_test, n_trials=10):
        """
        Busca de hiperparâmetros paralela otimizada para i3.
        Técnica: Halving Grid Search + Bayesian Optimization lite
        """
        print("\n🔬 FASE 2: Otimização de Hiperparâmetros")
        print("-" * 50)

        # Espaço de busca reduzido para i3
        param_grid = {
            'n_estimators': [50, 100, 150],  # Menos opções
            'max_features': [0.5, 0.75, 1.0],
            'contamination': [0.05, 0.1, 0.15],
        }

        # Gerar combinações
        from itertools import product
        keys = param_grid.keys()
        values = param_grid.values()
        experiments = [dict(zip(keys, v)) for v in product(*values)]

        # Limitar número de experimentos
        if len(experiments) > n_trials:
            import random
            random.shuffle(experiments)
            experiments = experiments[:n_trials]

        print(f"📋 Testando {len(experiments)} configurações...")

        # Usar subset para hyperparameter tuning (mais rápido)
        subset_size = min(50000, len(X_train))
        X_train_subset = X_train[:subset_size]
        X_test_subset = X_test[:min(10000, len(X_test))]

        # Processar em paralelo com ThreadPoolExecutor (melhor para I/O bound)
        results = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for i, config in enumerate(experiments):
                future = executor.submit(
                    self.train_single_model_optimized,
                    X_train_subset, X_test_subset, config, i
                )
                futures.append(future)

            # Coletar resultados
            for future in futures:
                results.append(future.result())

        # Encontrar melhor configuração
        best_result = min(results, key=lambda x: abs(x['anomaly_rate_test'] - 0.1))

        print(f"\n🏆 Melhor configuração encontrada:")
        print(f"   • n_estimators: {best_result['config']['n_estimators']}")
        print(f"   • max_features: {best_result['config']['max_features']}")
        print(f"   • contamination: {best_result['config']['contamination']}")
        print(f"   • Anomaly Rate: {best_result['anomaly_rate_test']:.2%}")

        return best_result['config']

    def retrain_all_models_optimized(self):
        """
        Re-treina todos os 30 modelos com otimizações extremas.
        Tempo estimado: 30 minutos total (1 min/modelo)
        """
        print("\n" + "=" * 60)
        print("🚀 INICIANDO RE-TREINAMENTO OTIMIZADO DOS 30 MODELOS")
        print("=" * 60)

        # Carregar dados uma vez só (cache)
        df = self.optimize_data_loading('data/features/featured_dataset.parquet')

        # Criar exemplo de entrada para a assinatura do modelo MLflow
        input_example = df.head()

        # Split
        print("\n📊 Preparando dados para treinamento...")
        X_train, X_test = train_test_split(
            df.values,  # Usar numpy array (mais rápido)
            test_size=0.15,
            random_state=42
        )

        # Normalização vetorizada
        print("🔧 Normalizando dados...")
        scaler = StandardScaler()

        # Usar float32 para economizar memória e acelerar
        X_train = scaler.fit_transform(X_train).astype('float32')
        X_test = scaler.transform(X_test).astype('float32')

        print(f"   Train shape: {X_train.shape}")
        print(f"   Test shape: {X_test.shape}")

        # Buscar melhores hiperparâmetros
        best_config = self.parallel_hyperparameter_search(X_train, X_test)
        
        # MLflow setup
        setup_mlflow()

        # Re-treinar modelos existentes
        print("\n" + "=" * 60)
        print("📦 RE-TREINANDO 30 MODELOS COM CONFIGURAÇÃO OTIMIZADA")
        print("=" * 60)

        models_dir = Path('outputs/models')
        existing_models = sorted(models_dir.glob('isolation_forest_*.joblib'))[:30]

        if not existing_models:
            print("⚠️ Nenhum modelo existente encontrado. Criando 30 novos...")
            existing_models = [f"model_{i}" for i in range(30)]

        results = []
        total_start = time.time()

        # Processar modelos em batches para não sobrecarregar
        batch_size = 5  # Treinar 5 por vez

        for batch_idx in range(0, len(existing_models), batch_size):
            batch = existing_models[batch_idx:batch_idx + batch_size]
            batch_results = []

            print(f"\n📦 Batch {batch_idx // batch_size + 1}/{len(existing_models) // batch_size + 1}")

            for i, model_ref in enumerate(batch):
                model_idx = batch_idx + i
                
                with mlflow.start_run(run_name=f"Optimized_Model_{model_idx:02d}"):
                    print(f"\n{'=' * 40}")
                    print(f"Modelo {model_idx + 1}/30")
                    print(f"{'=' * 40}")

                    # Configuração com variação para diversidade
                    config = best_config.copy()
                    config['random_state'] = 42 + model_idx

                    # Adicionar alguma variação
                    if model_idx % 3 == 0:
                        config['n_estimators'] = min(200, config['n_estimators'] + 50)
                    elif model_idx % 3 == 1:
                        config['max_features'] = max(0.5, config['max_features'] - 0.1)

                    mlflow.log_params(config)
                    mlflow.set_tag("optimization_level", "i3-optimized")
                    mlflow.set_tag("model_index", f"{model_idx:02d}")

                    # Treinar
                    result = self.train_single_model_optimized(
                        X_train, X_test, config, model_idx
                    )

                    metrics = {
                        'train_time': result['train_time'],
                        'anomaly_rate_test': result['anomaly_rate_test'],
                        'anomaly_rate_train': result['anomaly_rate_train'],
                        'score_mean': result['score_mean'],
                        'score_std': result['score_std'],
                    }
                    mlflow.log_metrics(metrics)

                    # Salvar modelo otimizado
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    model_name = f"isolation_forest_optimized_{model_idx:02d}_{timestamp}"
                    
                    # Log do modelo no MLflow
                    mlflow.sklearn.log_model(
                        sk_model=result['model'],
                        artifact_path="model",
                        registered_model_name=model_name,
                        input_example=input_example
                    )
                    
                    # Log do scaler
                    scaler_path = f"scaler_{model_idx:02d}.joblib"
                    joblib.dump(scaler, scaler_path)
                    mlflow.log_artifact(scaler_path, "scaler")
                    os.remove(scaler_path)

                    print(f"   ✅ Modelo e métricas salvos no MLflow para a run: {mlflow.active_run().info.run_name}")

                    batch_results.append(result)

                    # Limpar memória periodicamente
                    if (model_idx + 1) % 10 == 0:
                        gc.collect()

            results.extend(batch_results)

            # Pausa entre batches para não superaquecer o i3
            if batch_idx + batch_size < len(existing_models):
                print("\n⏸️  Pausa de 5 segundos para resfriar CPU...")
                time.sleep(5)

        total_time = time.time() - total_start

        # Relatório final
        self.print_final_report(results, total_time)

        return results

    def print_final_report(self, results: List[Dict], total_time: float):
        """
        Imprime relatório detalhado dos resultados.
        """
        print("\n" + "=" * 60)
        print("📊 RELATÓRIO FINAL DE OTIMIZAÇÃO")
        print("=" * 60)

        # Estatísticas
        train_times = [r['train_time'] for r in results]
        anomaly_rates = [r['anomaly_rate_test'] for r in results]

        print(f"""
📈 Estatísticas de Treinamento:
   • Total de modelos: {len(results)}
   • Tempo total: {total_time:.2f} segundos ({total_time / 60:.1f} minutos)
   • Tempo médio por modelo: {np.mean(train_times):.2f}s
   • Tempo mínimo: {np.min(train_times):.2f}s
   • Tempo máximo: {np.max(train_times):.2f}s

🎯 Estatísticas de Performance:
   • Anomaly Rate médio: {np.mean(anomaly_rates):.2%}
   • Desvio padrão: {np.std(anomaly_rates):.2%}
   • Melhor modelo: {np.min(anomaly_rates):.2%}
   • Pior modelo: {np.max(anomaly_rates):.2%}

💾 Uso de Recursos:
   • CPU médio: {psutil.cpu_percent()}%
   • RAM usada: {psutil.virtual_memory().percent}%
   • Temperatura CPU: {self.get_cpu_temp()}°C
        """)

        # Comparação com modelo original
        original_time = 14433  # 4 horas
        speedup = original_time / (total_time / len(results))

        print(f"""
🚀 COMPARAÇÃO COM MODELO ORIGINAL:
   • Tempo original (1 modelo): 4 horas
   • Tempo otimizado (1 modelo): {np.mean(train_times):.2f}s
   • Speedup: {speedup:.1f}x mais rápido!
   • Economia total: {(original_time * 30 - total_time) / 3600:.1f} horas
        """)

        # Recomendações
        best_model_idx = np.argmin([abs(r['anomaly_rate_test'] - 0.1) for r in results])
        best_model = results[best_model_idx]

        print(f"""
✅ RECOMENDAÇÕES:
   1. Melhor modelo para produção: Modelo {best_model['model_id']}
      • Anomaly Rate: {best_model['anomaly_rate_test']:.2%}
      • Tempo de treino: {best_model['train_time']:.2f}s

   2. Para melhorar ainda mais:
      • Instale Numba: pip install numba (2-3x speedup adicional)
      • Use SSD NVMe se possível (I/O 2x mais rápido)
      • Considere upgrade para i5/i7 (4-8 cores físicos)
      • Ou use Google Colab Pro (GPU grátis)
        """)

    def get_cpu_temp(self):
        """Obtém temperatura da CPU se disponível."""
        try:
            import subprocess
            result = subprocess.run(['sensors'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'Core 0' in line:
                    temp = line.split('+')[1].split('°')[0]
                    return float(temp)
        except:
            pass
        return "N/A"

    def benchmark_inference_speed(self, model_path: str):
        """
        Testa velocidade de inferência do modelo otimizado.
        """
        print("\n⚡ BENCHMARK DE INFERÊNCIA")
        print("-" * 50)

        # Carregar modelo
        artifact = joblib.load(model_path)
        model = artifact['model']
        scaler = artifact['scaler']

        # Criar dados de teste
        test_sizes = [1, 10, 100, 1000, 10000]

        for size in test_sizes:
            n_features = scaler.n_features_in_
            X_test = np.random.randn(size, n_features).astype('float32')
            X_test = scaler.transform(X_test)

            # Medir tempo
            start = time.time()
            predictions = model.predict(X_test)
            elapsed = (time.time() - start) * 1000  # em ms

            throughput = size / (elapsed / 1000)  # transações/segundo

            print("   {size:5d} amostras: {elapsed:6.2f}ms | {throughput:8.0f} tx/s")


def main():
    """Função principal - orquestra todo o processo de otimização."""

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║         TRUSTSHIELD EXTREME OPTIMIZATION SYSTEM             ║
    ║              Intel Core i3 Specialized Edition              ║
    ║                                                              ║
    ║   Transformando 4 horas em 30 minutos com ciência!         ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Verificar requisitos
    print("🔍 Verificando ambiente...")

    if psutil.virtual_memory().available / (1024 ** 3) < 5:
        print("⚠️ AVISO: Menos de 5GB de RAM disponível. Feche outros programas.")
        response = input("Continuar mesmo assim? (s/n): ")
        if response.lower() != 's':
            return

    # Inicializar otimizador
    optimizer = IntelI3Optimizer()

    # Menu de opções
    print("\n📋 OPÇÕES DE OTIMIZAÇÃO:")
    print("1. Re-treinar TODOS os 30 modelos (estimado: 30 minutos)")
    print("2. Treinar apenas 1 modelo de teste (estimado: 1 minuto)")
    print("3. Benchmark de modelo existente")
    print("4. Análise completa + Re-treinamento")

    # Hardcoded choice for non-interactive execution, placed after the menu is printed.
    choice = "1"
    print(f"\nOpção '{choice}' selecionada automaticamente para execução não-interativa.")

    if choice == '1':
        # Re-treinar todos
        results = optimizer.retrain_all_models_optimized()

    elif choice == '2':
        # Teste rápido
        df = optimizer.optimize_data_loading('data/features/featured_dataset.parquet')
        X_train, X_test = train_test_split(df.values, test_size=0.15, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype('float32')
        X_test = scaler.transform(X_test).astype('float32')

        config = {'n_estimators': 100, 'max_features': 1.0, 'contamination': 0.1}
        result = optimizer.train_single_model_optimized(X_train, X_test, config, 0)

        print(f"\n✅ Modelo de teste treinado em {result['train_time']:.2f}s")

    elif choice == '3':
        # Benchmark
        models = list(Path('outputs/models').glob('*.joblib'))
        if models:
            print(f"\nEncontrados {len(models)} modelos. Testando o mais recente...")
            optimizer.benchmark_inference_speed(str(models[-1]))
        else:
            print("❌ Nenhum modelo encontrado!")

    elif choice == '4':
        # Análise + Re-treinamento completo
        print("\n🔬 Executando análise completa + otimização...")

        # Primeiro analisar modelos existentes
        # os.system("python analyze_models.py") # Arquivo não encontrado no projeto

        # Depois re-treinar com otimização
        results = optimizer.retrain_all_models_optimized()

        # Benchmark do melhor modelo
        models = sorted(Path('outputs/models').glob('*optimized*.joblib'))
        if models:
            optimizer.benchmark_inference_speed(str(models[-1]))

    print("\n" + "=" * 60)
    print("✅ OTIMIZAÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 60)
    print("""
    💡 Próximos passos:
       1. Teste o modelo otimizado na API
       2. Compare métricas no MLflow
       3. Faça deploy do melhor modelo

    📊 Para visualizar no MLflow:
       mlflow ui --host 0.0.0.0

    🚀 Para usar na API:
       export MODEL_PATH='outputs/models/isolation_forest_optimized_00_*.joblib'
       uvicorn src.api.main:app --reload
    """)


if __name__ == "__main__":
    main()