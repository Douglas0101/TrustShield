# ==============================================================================
# ARQUIVO: src/models/validation.py (OTIMIZADO)
# ==============================================================================
# -*- coding: utf-8 -*-
"""
Módulo de Validação do Projeto TrustShield
Versão: 3.0.0 - High Performance & Robustness

Responsabilidades:
- Validar a estrutura e os valores do arquivo de configuração (config.yaml).
- Validar o schema, a qualidade e as regras de negócio dos dataframes.
"""
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import pandera as pa

# --- Configuração do Logger ---
def get_logger(name: str) -> logging.Logger:
    """Configura e retorna um logger padrão."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - [TrustShield-Validation] - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

LOGGER = get_logger(__name__)


# --- Validação da Configuração ---
class ConfigValidator:
    """Valida a estrutura e os valores do arquivo de configuração."""

    @staticmethod
    def validate(config: Dict[str, Any]) -> None:
        """
        Executa uma série de verificações no dicionário de configuração.

        Args:
            config: O dicionário de configuração carregado do YAML.

        Raises:
            ValueError: Se uma chave obrigatória estiver em falta ou um valor for inválido.
        """
        LOGGER.info("Validando arquivo de configuração...")

        # Configurações do projeto
        if 'project' not in config or 'random_state' not in config['project']:
            raise ValueError("Erro de config: 'project.random_state' está em falta.")

        # Caminhos de dados
        if 'data' not in config or 'featured_filename' not in config['data']:
            raise ValueError("Erro de config: 'data.featured_filename' está em falta.")

        # Configurações do modelo
        if 'model' not in config or 'default_params' not in config['model']:
            raise ValueError("Erro de config: 'model.default_params' está em falta.")

        # Configurações de HPO (se habilitado)
        if config.get('hyper_optimization', {}).get('enabled', False):
            hpo_config = config['hyper_optimization']
            if 'library' not in hpo_config or 'space' not in hpo_config:
                raise ValueError("Erro de config: HPO está habilitado mas 'library' ou 'space' está em falta.")
            # Verifica os espaços de busca de hiperparâmetros obrigatórios
            for param in ['n_estimators', 'max_samples', 'max_features']:
                if param not in hpo_config['space']:
                    raise ValueError(f"Erro de config: Espaço de busca de HPO está em falta para '{param}'.")

        LOGGER.info("Arquivo de configuração validado com sucesso.")


# --- Validação de Dados ---
class DataValidator:
    """
    Lida com a validação de schema, regras de negócio e qualidade dos dados para dataframes.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_features = config['data']['model_features']

    def validate_schema(self, df: pd.DataFrame) -> None:
        """
        Valida o dataframe contra um schema Pandera construído dinamicamente.

        Esta versão corrigida aplica regras de validação específicas e lógicas para
        colunas conhecidas como 'birth_year' e 'amount'.

        Args:
            df: O dataframe a ser validado.

        Raises:
            pandera.errors.SchemaErrors: Se o dataframe falhar na validação.
        """
        LOGGER.info("Validando schema dos dados...")
        schema_map = {}
        current_year = datetime.now().year

        for col_name in self.model_features:
            if col_name not in df.columns:
                LOGGER.warning(f"Feature '{col_name}' do config não encontrada nos dados. A ignorar na validação.")
                continue

            # Regra específica para ano de nascimento
            if 'birth_year' in col_name:
                schema_map[col_name] = pa.Column(
                    float,
                    nullable=False,
                    checks=pa.Check.in_range(1920, current_year, error=f"Valor inválido para {col_name}")
                )
            # Regra específica para valores monetários
            elif 'amount' in col_name:
                schema_map[col_name] = pa.Column(
                    float,
                    nullable=False,
                    # Um intervalo generoso para detetar outliers extremos
                    checks=pa.Check.in_range(-100000, 100000, error=f"Valor não razoável para {col_name}")
                )
            # Regra genérica para outras colunas numéricas
            else:
                schema_map[col_name] = pa.Column(float, nullable=False)

        if not schema_map:
            raise ValueError("Não foi possível construir o schema de validação. Nenhuma feature do modelo encontrada no dataframe.")

        # Criar e validar o schema
        schema = pa.DataFrameSchema(schema_map, ordered=False, coerce=True)
        try:
            schema.validate(df, lazy=True)
            LOGGER.info("Validação de schema dos dados concluída com sucesso.")
        except pa.errors.SchemaErrors as err:
            LOGGER.error(f"Validação de schema falhou com {len(err.failure_cases)} erros.")
            # Log detalhado dos casos de falha para facilitar a depuração
            LOGGER.error(f"Casos de falha:\n{err.failure_cases}")
            raise