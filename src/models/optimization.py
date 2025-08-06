# ==============================================================================
# ARQUIVO: src/models/optimization.py (OTIMIZADO)
# ==============================================================================
# -*- coding: utf-8 -*-
"""
Módulo de Otimização de Hiperparâmetros do Projeto TrustShield
Versão: 3.0.0 - High Performance & Robustness

Responsabilidades:
- Orquestrar o processo de otimização de hiperparâmetros (HPO).
- Implementar lógicas de otimização usando Optuna.
- Otimizar a performance do HPO através de pré-cálculo de transformações.
"""
import logging
import sys
from typing import Any, Dict

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# --- Configuração do Logger ---
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - [TrustShield-Optimizer] - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


LOGGER = get_logger(__name__)


# --- Lógica Base do Otimizador ---
class BaseOptimizer:
    """Classe base para otimizadores de hiperparâmetros."""

    def __init__(self, config: Dict[str, Any], X_train_scaled: np.ndarray, X_val_scaled: np.ndarray):
        self.config = config
        self.X_train_scaled = X_train_scaled
        self.X_val_scaled = X_val_scaled
        self.hpo_config = config['hyper_optimization']
        self.space_config = self.hpo_config['space']
        self.random_state = config['project']['random_state']

    def _objective_function(self, params: Dict[str, Any]) -> float:
        """
        Treina um modelo com os parâmetros dados e retorna a pontuação do objetivo.
        Esta é a função principal a ser minimizada/maximizada.
        """
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)

            model = IsolationForest(**params).fit(self.X_train_scaled)
            scores_val = -model.decision_function(self.X_val_scaled)

            # Objetivo: Maximizar a variância dos scores de anomalia para obter melhor separação
            objective_value = float(np.var(scores_val))
            mlflow.log_metric("hpo_objective_variance", objective_value)

            return objective_value


# --- Implementação com Optuna ---
class OptunaOptimizer(BaseOptimizer):
    """Otimizador baseado em Optuna."""

    def optimize(self) -> Dict[str, Any]:
        LOGGER.info("Iniciando HPO com Optuna...")

        def objective(trial: optuna.trial.Trial) -> float:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', *self.space_config['n_estimators']),
                'max_samples': trial.suggest_float('max_samples', *self.space_config['max_samples']),
                'max_features': trial.suggest_float('max_features', *self.space_config['max_features']),
                'contamination': self.config['model']['default_params'].get('contamination', 'auto'),
                'n_jobs': -1,
                'random_state': self.random_state
            }
            return self._objective_function(params)

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=self.hpo_config.get('n_trials', 50))

        LOGGER.info(f"Otimização com Optuna concluída. Melhores parâmetros: {study.best_params}")
        return study.best_params


# --- Orquestrador ---
class HybridOptimizer:
    """
    Orquestra o processo de HPO selecionando a biblioteca especificada no config
    e preparando os dados para evitar recomputação dentro do loop de otimização.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.library = config.get('hyper_optimization', {}).get('library', 'optuna')

    def optimize(self, df_train: pd.DataFrame, df_val: pd.DataFrame) -> Dict[str, Any]:
        """
        Prepara os dados e executa a biblioteca de otimização selecionada.

        Args:
            df_train: Dataframe de treino.
            df_val: Dataframe de validação.

        Returns:
            Um dicionário com os melhores hiperparâmetros encontrados.
        """
        LOGGER.info(f"Preparando dados para HPO (Biblioteca: {self.library})...")

        # --- OTIMIZAÇÃO DE PERFORMANCE ---
        # Normaliza os dados UMA VEZ antes do loop de otimização para evitar overhead massivo.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(df_train)
        X_val_scaled = scaler.transform(df_val)

        LOGGER.info("Dados normalizados. Iniciando loop de otimização...")

        if self.library == 'optuna':
            optimizer = OptunaOptimizer(self.config, X_train_scaled, X_val_scaled)
            return optimizer.optimize()
        # Adicionar 'hyperopt' ou outras bibliotecas aqui se necessário
        # elif self.library == 'hyperopt':
        #     ...
        else:
            raise ValueError(f"Biblioteca de HPO não suportada: {self.library}")