# src/api/main.py

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Annotated

from fastapi import FastAPI, HTTPException, Request, Depends
from starlette.datastructures import State

# Importa a sua classe de predição principal
from src.models.predict import TrustShieldPredictor

# --- Configuração de Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("trustshield_api")


# --- Estado Tipado (Typed State) ---
# Esta é a solução sofisticada para o aviso do linter.
# Criamos uma classe que define a "forma" do nosso estado,
# permitindo que o linter e o autocompletar entendam app.state.
class AppState(State):
    predictor: TrustShieldPredictor | None


# --- Gerenciador de Ciclo de Vida (Lifespan) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação. Carrega o modelo na inicialização
    e o anexa a um estado tipado.
    """
    logger.info("Iniciando ciclo de vida da API...")

    # Inicializa o estado da aplicação com nossa classe tipada.
    # Isso resolve o aviso "Unresolved attribute reference 'state'".
    app.state = AppState()

    try:
        app.state.predictor = TrustShieldPredictor()
        logger.info("Motor de inferência carregado com sucesso no estado da aplicação.")
    except Exception as e:
        logger.critical(f"Falha crítica ao carregar o modelo na inicialização: {e}", exc_info=True)
        app.state.predictor = None

    yield  # A aplicação FastAPI é executada aqui.

    # --- Código de Limpeza (executado no shutdown) ---
    logger.info("Encerrando ciclo de vida da API e limpando recursos...")
    app.state.predictor = None


# --- Instância da Aplicação FastAPI ---
app = FastAPI(
    title="TrustShield Fraud Detection API",
    description="API de produção para detecção de anomalias em tempo real, construída com as melhores práticas.",
    version="2.1.0-typed",
    lifespan=lifespan
)


# --- Injeção de Dependência Refinada ---
async def get_predictor(request: Request) -> TrustShieldPredictor:
    """
    Dependência que retorna a instância do motor de inferência a partir do estado.
    Levanta uma exceção se o modelo não estiver disponível.
    """
    # Acessar o estado através do 'request' é a prática recomendada dentro de endpoints.
    # O linter agora entende 'request.state.predictor' por causa da nossa classe AppState.
    predictor = request.app.state.predictor
    if not predictor:
        raise HTTPException(
            status_code=503,  # Service Unavailable
            detail="Serviço indisponível: O modelo de predição não está carregado."
        )
    return predictor


# Define um tipo anotado para a dependência, melhorando a legibilidade.
PredictorDep = Annotated[TrustShieldPredictor, Depends(get_predictor)]


# --- Endpoints da API ---

@app.post("/predict", tags=["Prediction"])
async def predict_transaction(
        input_data: Dict[str, Any],
        predictor: PredictorDep
):
    """
    Endpoint para realizar a predição de anomalia para uma transação.
    """
    try:
        result = predictor.predict(input_data)
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=f"Erro interno na predição: {result.get('error')}")
        return result
    except Exception as e:
        logger.error(f"Erro inesperado ao processar requisição /predict: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro inesperado no servidor: {e}")


@app.get("/status", tags=["Health Check"])
async def get_status(
        predictor: PredictorDep
):
    """
    Endpoint de health check que verifica o status do serviço e do modelo.
    """
    return predictor.get_status()