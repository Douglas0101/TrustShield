# -*- coding: utf-8 -*-
"""
Módulo de Engenharia de Features para o Projeto TrustShield.

Responsabilidades:
1. Carregar o dataset processado (saída de make_dataset.py).
2. Aplicar a limpeza de dados (tipos, valores monetários).
3. Criar novas features temporais e comportamentais.
4. Salvar o dataset final enriquecido, pronto para a modelagem.
"""
import logging
import sys
from pathlib import Path

import pandas as pd


# --- Configuração do Logger ---
def get_logger(name: str) -> logging.Logger:
    """Configura e retorna um logger simples e consistente."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - [TrustShield] - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


# --- Funções de Feature Engineering ---
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica a limpeza e conversão de tipos de dados."""
    logger = get_logger(__name__)
    logger.info("Iniciando limpeza de dados...")

    def clean_money(column):
        return pd.to_numeric(column.astype(str).str.replace('$', ''), errors='coerce')

    money_columns = ['amount', 'per_capita_income', 'yearly_income', 'total_debt']
    for col in money_columns:
        if col in df.columns:
            df[col] = clean_money(df[col])

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    logger.info("Limpeza de dados concluída.")
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria novas features a partir dos dados existentes."""
    logger = get_logger(__name__)
    logger.info("Iniciando engenharia de features...")

    df['transaction_hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    df['is_night_transaction'] = (df['transaction_hour'] <= 6) | (df['transaction_hour'] >= 22)

    avg_amount_per_user = df.groupby('client_id')['amount'].transform('mean')
    df['amount_vs_avg'] = df['amount'] / (avg_amount_per_user + 1)

    logger.info("Engenharia de features concluída.")
    return df


# --- Orquestrador Principal ---
def main():
    """Orquestra a execução do pipeline de engenharia de features."""
    logger = get_logger(__name__)
    logger.info("🚀 Pipeline de engenharia de features iniciado.")

    # ALINHADO COM O DIRETÓRIO: Define a raiz do projeto de forma robusta.
    project_root = Path(__file__).resolve().parents[2]

    # Define os caminhos com base na raiz.
    processed_dir = project_root / "data" / "processed"
    feature_dir = project_root / "data" / "features"

    input_file = processed_dir / "primary_dataset.parquet"
    output_file = feature_dir / "featured_dataset.parquet"

    # Garante que o diretório de saída exista.
    feature_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Carregando dados de {input_file}")
        df = pd.read_parquet(input_file)

        df_clean = clean_data(df)
        df_featured = create_features(df_clean)

        logger.info(f"Salvando dataset com features em {output_file}")
        df_featured.to_parquet(output_file, index=False)

        logger.info("✅ Pipeline de engenharia de features concluído com sucesso.")
    except FileNotFoundError:
        logger.error(f"Erro: Ficheiro de entrada não encontrado em {input_file}. Execute 'make_dataset.py' primeiro.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Um erro inesperado ocorreu: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()