# ==============================================================================
# ARQUIVO: src/models/interpretation.py (OTIMIZADO)
# ==============================================================================
# -*- coding: utf-8 -*-
"""
Módulo de Interpretabilidade do Projeto TrustShield
Versão: 3.0.0 - High Performance & Robustness

Responsabilidades:
- Fornecer uma arquitetura extensível para interpretar modelos.
- Implementar métodos de interpretação (SHAP).
- Otimizar a performance usando explicadores específicos de modelo.
- Garantir a correta serialização dos resultados.
"""
import json
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Type

import numpy as np
import pandas as pd
import shap


# --- Configuração do Logger ---
def get_logger(name: str) -> logging.Logger:
    """Configura e retorna um logger padrão."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - [TrustShield-Interpreter] - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


LOGGER = get_logger(__name__)


# --- Classe Base Abstrata para Interpretação ---
class InterpretationMethod(ABC):
    """
    Classe base abstrata que define o contrato para todos os métodos de interpretação.
    """

    def __init__(self, model: Any, data: pd.DataFrame, config: Dict[str, Any], logger: logging.Logger):
        self.model = model
        self.data = data
        self.config = config
        self.logger = logger

    @abstractmethod
    def explain(self) -> Dict[str, Any]:
        """Executa a lógica principal do método de interpretação."""
        pass

    @abstractmethod
    def save_results(self, results: Dict[str, Any], path: str) -> None:
        """Salva os resultados gerados pelo método 'explain'."""
        pass


# --- Implementação com SHAP ---
class ShapInterpreter(InterpretationMethod):
    """
    Implementação de interpretação com SHAP, otimizada para modelos baseados em árvore.
    """

    def __init__(self, model: Any, data: pd.DataFrame, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(model, data, config, logger)
        self.max_samples = self.config.get('interpretability', {}).get('shap', {}).get('max_samples', 1000)

    def explain(self) -> Dict[str, Any]:
        """Calcula os valores SHAP para uma amostra dos dados."""
        self.logger.info(f"Iniciando análise SHAP com até {self.max_samples} amostras...")
        try:
            sample_data = self.data.sample(n=min(len(self.data), self.max_samples), random_state=42)

            # --- OTIMIZAÇÃO DE PERFORMANCE ---
            # Usa TreeExplainer para modelos baseados em árvore como o IsolationForest.
            # É significativamente mais rápido e preciso que o KernelExplainer.
            explainer = shap.TreeExplainer(self.model, sample_data)
            shap_values = explainer.shap_values(sample_data)

            feature_importance = dict(zip(sample_data.columns, np.abs(shap_values).mean(axis=0)))

            self.logger.info("Análise SHAP concluída com sucesso.")
            return {
                "base_values": explainer.expected_value,
                "values": shap_values,
                "feature_importance": feature_importance
            }
        except Exception as e:
            self.logger.error(f"Ocorreu um erro durante a análise SHAP: {e}", exc_info=True)
            return {}

    def save_results(self, results: Dict[str, Any], path: str) -> None:
        """Salva os resultados SHAP num ficheiro JSON, garantindo a serialização."""
        if not results:
            self.logger.warning("Nenhum resultado SHAP para salvar.")
            return

        try:
            # --- CORREÇÃO DE BUG ---
            # Converte arrays numpy para listas para serem compatíveis com JSON.
            serializable_results = {
                'base_values': float(results['base_values']),
                'values': results['values'].tolist() if isinstance(results.get('values'), np.ndarray) else results.get(
                    'values'),
                'feature_importance': {k: float(v) for k, v in results.get('feature_importance', {}).items()}
            }

            output_path = Path(path)
            output_path.mkdir(parents=True, exist_ok=True)
            filepath = output_path / "shap_values.json"

            with open(filepath, "w") as f:
                json.dump(serializable_results, f, indent=2)

            self.logger.info(f"Resultados SHAP salvos em {filepath}")
        except Exception as e:
            self.logger.error(f"Falha ao salvar resultados SHAP: {e}", exc_info=True)


# --- Orquestrador ---
class ModelInterpreter:
    """Orquestra e executa os diferentes métodos de interpretação."""

    def __init__(self, model: Any, data: pd.DataFrame, config: Dict[str, Any]):
        self.model = model
        self.data = data
        self.config = config
        self.logger = get_logger("ModelInterpreter")
        self.methods: Dict[str, Type[InterpretationMethod]] = {
            "shap": ShapInterpreter,
        }

    def run_and_save(self, output_path: str) -> None:
        """Executa todos os métodos de interpretação habilitados e salva os seus resultados."""
        enabled_methods = self.config.get('interpretability', {}).get('methods', [])
        self.logger.info(f"Métodos de interpretação habilitados: {enabled_methods}")

        for method_name in enabled_methods:
            if method_name in self.methods:
                self.logger.info(f"--- Executando método: {method_name.upper()} ---")

                interpreter_class = self.methods[method_name]
                interpreter_instance = interpreter_class(self.model, self.data, self.config, self.logger)

                results = interpreter_instance.explain()
                if results:
                    method_output_path = Path(output_path) / method_name
                    interpreter_instance.save_results(results, str(method_output_path))
            else:
                self.logger.warning(f"Método de interpretação '{method_name}' não reconhecido. A ignorar.")