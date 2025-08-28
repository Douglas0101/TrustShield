# -*- coding: utf-8 -*-
"""
TrustShield API (FastAPI) — versão otimizada e robusta
Integração explícita com mlflow_setup para garantir a conexão correta (HTTP).
"""

# =============================================================================
# Imports & setup
# =============================================================================
import logging
import os
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field

# --- CORREÇÃO ESTRUTURAL ---
# Importa a função de setup diretamente do seu utilitário.
from src.utils.mlflow_setup import setup_mlflow

# Configuração do Logger (mantida da sua versão original)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [TrustShield-API] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Estado da Aplicação e Carregamento do Modelo
# =============================================================================
class AppState:
    """Mantém o estado da aplicação, como o modelo carregado."""

    def __init__(self):
        self.model = None
        self.model_path = os.getenv("MODEL_PATH", "outputs/models/default_model.joblib")
        self.mlflow_client = None


app_state = AppState()


async def load_model_and_setup_mlflow():
    """
    Função assíncrona para carregar o modelo e configurar o MLflow.
    Executada durante o startup da API.
    """
    global app_state
    logger.info("Iniciando configuração do MLflow e carregamento do modelo...")

    try:
        # --- CORREÇÃO APLICADA AQUI ---
        # Chama a sua função de setup para configurar as URIs corretas (HTTP)
        # antes de qualquer outra operação do MLflow.
        mlflow = setup_mlflow(experiment="TrustShield_API")
        app_state.mlflow_client = mlflow.tracking.MlflowClient()
        logger.info(f"MLflow configurado com sucesso. Tracking URI: {mlflow.get_tracking_uri()}")

        # Tenta registrar um "run" de inicialização para validar a conexão.
        max_retries = 10
        for attempt in range(max_retries):
            try:
                with mlflow.start_run(run_name="api_startup"):
                    mlflow.set_tag("status", "API started")
                logger.info("Run de inicialização no MLflow registrado com sucesso.")
                break
            except Exception as e:
                logger.warning(
                    f"Tentativa {attempt + 1}/{max_retries} de conectar ao MLflow falhou. Aguardando... Erro: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(5)
                else:
                    logger.error("Falha crítica ao conectar ao MLflow após múltiplas tentativas.")
                    # A API continuará, mas sem rastreamento MLflow.
                    app_state.mlflow_client = None

    except Exception as e:
        logger.error(f"Erro inesperado durante o setup do MLflow: {e}", exc_info=True)

    # Carregamento do modelo (lógica original mantida)
    model_file = Path(app_state.model_path)
    if model_file.exists():
        try:
            app_state.model = joblib.load(model_file)
            logger.info(f"Modelo '{model_file.name}' carregado com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao carregar o modelo de {model_file}: {e}", exc_info=True)
            app_state.model = None
    else:
        logger.warning(f"Arquivo do modelo não encontrado em {model_file}. API iniciará sem modelo.")
        app_state.model = None


# =============================================================================
# Ciclo de Vida da API (Lifespan)
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o startup e shutdown da aplicação."""
    await load_model_and_setup_mlflow()
    yield
    logger.info("API encerrada.")


app = FastAPI(lifespan=lifespan)


# =============================================================================
# DTOs (Data Transfer Objects)
# =============================================================================
class TransactionInput(BaseModel):
    """Estrutura dos dados de entrada para uma predição."""
    amount: float = Field(..., example=123.45)

    # Adicione aqui os outros campos que seu modelo espera
    # Exemplo: 'hour', 'day_of_week', etc.
    class Config:
        extra = 'allow'  # Permite campos extras para flexibilidade


class PredictionOutput(BaseModel):
    """Estrutura da resposta da predição."""
    is_anomaly: int = Field(..., example=1, description="-1 para anomalia (fraude), 1 para normal.")
    score: float = Field(..., example=-0.2345)
    model_version: str


# =============================================================================
# Endpoints da API
# =============================================================================
@app.get("/healthz", tags=["Health"])
def health_check():
    """Endpoint para verificação de saúde (liveness probes)."""
    return {"status": "ok"}


@app.get("/status", tags=["Health"])
def get_status():
    """Retorna o status atual do modelo carregado."""
    return {
        "model_loaded": app_state.model is not None,
        "model_path": app_state.model_path,
        "mlflow_connected": app_state.mlflow_client is not None
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(transaction: TransactionInput):
    """Executa a predição de anomalia para uma transação."""
    if app_state.model is None:
        raise HTTPException(status_code=503, detail="Modelo não está disponível no momento.")

    try:
        # Converte o input Pydantic para um DataFrame do Pandas
        input_df = pd.DataFrame([transaction.dict()])

        # ATENÇÃO: Garanta que as features no input_df correspondam
        # exatamente às features esperadas pelo modelo.
        # A ordem e os nomes das colunas são cruciais.

        # Exemplo: Selecionar apenas as colunas que o modelo usa
        # feature_columns = ['amount', 'hour', ...]
        # input_features = input_df[feature_columns]

        # Para o IsolationForest, geralmente se usam todas as features numéricas.
        # Esta é uma suposição que precisa ser validada com seu script de treino.
        input_features = input_df.select_dtypes(include=['number'])

        score = app_state.model.decision_function(input_features)[0]
        prediction = app_state.model.predict(input_features)[0]

        return {
            "is_anomaly": int(prediction),
            "score": float(score),
            "model_version": Path(app_state.model_path).name
        }
    except Exception as e:
        logger.error(f"Erro durante a predição: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro interno no servidor: {e}")


@app.post("/reload", tags=["Admin"])
async def reload_model(background_tasks: BackgroundTasks):
    """Dispara a recarga do modelo e a reconfiguração do MLflow em background."""
    logger.info("Requisição para recarregar o modelo recebida.")
    background_tasks.add_task(load_model_and_setup_mlflow)
    return {"message": "Processo de recarga do modelo iniciado."}