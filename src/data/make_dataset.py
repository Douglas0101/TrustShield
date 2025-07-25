# -*- coding: utf-8 -*-
"""
Módulo de Ingestão e Processamento de Dados para o Projeto TrustShield.

Responsabilidades:
1. Carregar os dados brutos de transações, usuários e códigos MCC.
2. Validar a integridade e o schema básico dos dados de entrada.
3. Mesclar os datasets para criar uma visão consolidada.
4. Salvar o dataset processado em um formato otimizado (Parquet) para as próximas etapas.

Execução via linha de comando:
    python src/data/make_dataset.py
"""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


class JsonFormatter(logging.Formatter):
    """Formatador para logs estruturados em JSON."""
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.name,
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


def get_logger(name: str) -> logging.Logger:
    """Configura e retorna um logger com formato JSON."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger


def load_raw_data(
    input_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Carrega os datasets brutos do diretório de entrada."""
    logger = get_logger(__name__)
    logger.info(f"Iniciando carregamento de dados do diretório: {input_path}")

    transactions_file = input_path / "transactions_data.csv"
    users_file = input_path / "users_data.csv"
    mcc_file = input_path / "mcc_codes.json"

    if not all([transactions_file.exists(), users_file.exists(), mcc_file.exists()]):
        logger.error("Arquivos de dados essenciais não encontrados. Verifique o diretório 'data/raw'.")
        raise FileNotFoundError("Um ou mais arquivos de dados não foram encontrados.")

    transactions_df = pd.read_csv(transactions_file)
    users_df = pd.read_csv(users_file)
    with open(mcc_file, "r", encoding="utf-8") as f:
        mcc_map = json.load(f)

    logger.info(f"Dados carregados com sucesso: "
                f"{len(transactions_df)} transações, "
                f"{len(users_df)} usuários, "
                f"{len(mcc_map)} códigos MCC.")
    return transactions_df, users_df, mcc_map


def validate_data(transactions_df: pd.DataFrame, users_df: pd.DataFrame):
    """Valida a presença de colunas chave para a mesclagem."""
    logger = get_logger(__name__)
    logger.info("Iniciando validação de schema dos dados.")

    # --- CORREÇÃO APLICADA AQUI ---
    required_cols_transactions = {"client_id", "mcc"}
    required_cols_users = {"id"}
    # --- FIM DA CORREÇÃO ---

    if not required_cols_transactions.issubset(transactions_df.columns):
        raise ValueError(f"DataFrame de transações não contém as colunas necessárias: {required_cols_transactions}")
    if not required_cols_users.issubset(users_df.columns):
        raise ValueError(f"DataFrame de usuários não contém as colunas necessárias: {required_cols_users}")

    logger.info("Validação de schema concluída com sucesso.")


def merge_datasets(
    transactions_df: pd.DataFrame, users_df: pd.DataFrame
) -> pd.DataFrame:
    """Mescla os DataFrames de transações e usuários."""
    logger = get_logger(__name__)
    logger.info("Iniciando a mesclagem dos datasets de transações e usuários.")

    # --- CORREÇÃO APLICADA AQUI ---
    # Usa left_on e right_on porque os nomes das colunas de ID são diferentes
    merged_df = pd.merge(
        transactions_df,
        users_df,
        left_on="client_id",
        right_on="id",
        how="inner",
        suffixes=("_transaction", "_user"), # Renomeia colunas 'id' conflitantes
    )
    # Remove a coluna de id do usuário que se tornou redundante após a mesclagem
    merged_df = merged_df.drop(columns=["id_user"])
    # --- FIM DA CORREÇÃO ---

    logger.info(f"Mesclagem concluída. Dataset final possui {len(merged_df)} registros.")
    return merged_df


def save_processed_data(df: pd.DataFrame, output_path: Path):
    """Salva o DataFrame processado em formato Parquet."""
    logger = get_logger(__name__)
    logger.info(f"Salvando dataset processado em: {output_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "primary_dataset.parquet"

    df.to_parquet(output_file, index=False)

    logger.info(f"Dataset salvo com sucesso em {output_file}.")


def main():
    """Orquestra a execução do pipeline de ingestão e processamento de dados."""
    logger = get_logger(__name__)
    logger.info("Pipeline de criação de dataset iniciado.")

    project_root = Path(__file__).resolve().parents[2]
    input_dir = project_root / "data" / "raw"
    output_dir = project_root / "data" / "processed"

    try:
        transactions_df, users_df, _ = load_raw_data(input_dir)
        validate_data(transactions_df, users_df)
        processed_df = merge_datasets(transactions_df, users_df)
        save_processed_data(processed_df, output_dir)
        logger.info("Pipeline de criação de dataset concluído com sucesso.")

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Falha na execução do pipeline: {e}", exc_info=False)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Um erro inesperado ocorreu: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()