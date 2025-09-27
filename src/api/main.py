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
from typing import Any, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Setup do MLflow (utilitário do seu projeto)
from src.utils.mlflow_setup import setup_mlflow
from src.api.security import enforce_ip_whitelist

# Configuração do Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [TrustShield-API] - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Estado da Aplicação e Carregamento do Modelo
# =============================================================================
class AppState:
    """Mantém o estado da aplicação, como o modelo carregado."""

    def __init__(self):
        self.model = None
        self.model_path = os.getenv(
            "MODEL_PATH",
            "outputs/models/default_model.joblib",
        )
        self.mlflow_client = None
        self.model_artifact = None
        self.data_transformer = None
        self.model_features: Optional[List[str]] = None
        self.model_structure = "unknown"

app_state = AppState()

def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    """Recupera atributos ignorando valores mockados em testes."""
    try:
        value = getattr(obj, name)
    except AttributeError:
        return default
    if value is None:
        return None
    module_name = getattr(value.__class__, "__module__", "")
    if module_name.startswith("unittest.mock"):
        return default
    return value

def _ensure_feature_list(raw_features: Any) -> Optional[List[str]]:
    """Normaliza a lista de features vinda do artefato do modelo."""
    if raw_features is None:
        return None
    if isinstance(raw_features, list):
        return raw_features
    if isinstance(raw_features, tuple):
        return list(raw_features)
    if hasattr(raw_features, "tolist"):
        return list(raw_features.tolist())
    if isinstance(raw_features, set):
        return list(raw_features)
    return None

def _prepare_model_components(artifact: Any) -> dict:
    """Extrai componentes relevantes de diferentes formatos de artefato."""
    model_obj = None
    transformer = None
    features = None
    structure = "unknown"

    if isinstance(artifact, dict):
        structure = "dict"
        for key in ("model", "pipeline", "estimator"):
            if artifact.get(key) is not None:
                model_obj = artifact[key]
                structure = f"dict:{key}"
                break

        if model_obj is None:
            for value in artifact.values():
                if hasattr(value, "predict"):
                    model_obj = value
                    structure = f"dict:{value.__class__.__name__}"
                    break

        transformer = (
            artifact.get("preprocessor")
            or artifact.get("transformer")
            or artifact.get("scaler")
        )
        features = artifact.get("features")
    else:
        model_obj = artifact
        structure = artifact.__class__.__name__
        features = getattr(artifact, "feature_names_in_", None)

    return {
        "model": model_obj,
        "transformer": transformer,
        "features": _ensure_feature_list(features),
        "structure": structure,
    }

def _resolve_runtime_components(state: AppState) -> tuple:
    """Resolve o modelo, transformador e features em tempo de execução."""
    model_container = _safe_getattr(state, "model")
    transformer = _safe_getattr(state, "data_transformer")
    features = _safe_getattr(state, "model_features")

    if isinstance(model_container, dict):
        components = _prepare_model_components(model_container)
        model = components["model"]
        transformer = transformer or components["transformer"]
        features = features or components["features"]
    else:
        model = model_container
    return model, transformer, features

async def load_model_and_setup_mlflow():
    """
    Carrega o modelo e configura o MLflow.
    Executada durante o startup da API.
    """
    global app_state
    logger.info("Iniciando configuração do MLflow e carregamento do modelo...")

    # ---- Setup do MLflow com retries ----
    try:
        mlflow = setup_mlflow(experiment="TrustShield_API")
        app_state.mlflow_client = mlflow.tracking.MlflowClient()
        logger.info(f"MLflow configurado. Tracking URI: {mlflow.get_tracking_uri()}")

        max_retries = 10
        for attempt in range(max_retries):
            try:
                with mlflow.start_run(run_name="api_startup"):
                    mlflow.set_tag("status", "API started")
                logger.info("Run de inicialização no MLflow registrado.")
                break
            except Exception as e:
                logger.warning(
                    f"Tentativa {attempt + 1}/{max_retries} MLflow falhou. Erro: {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(5)
                else:
                    logger.error("Falha ao conectar ao MLflow após múltiplas tentativas.")
                    app_state.mlflow_client = None
    except Exception as e:
        logger.error(f"Erro no setup do MLflow: {e}", exc_info=True)

    # ---- Carregamento do modelo ----
    model_file = Path(app_state.model_path)
    if model_file.exists():
        try:
            loaded_artifact = joblib.load(model_file)
            components = _prepare_model_components(loaded_artifact)

            if components["model"] is None:
                raise ValueError("Artefato de modelo não contém objeto de predição válido.")

            app_state.model = components["model"]
            app_state.model_artifact = loaded_artifact
            app_state.data_transformer = components["transformer"]
            app_state.model_features = components["features"]
            app_state.model_structure = components["structure"]

            feature_list = app_state.model_features or []
            transformer_name = (
                type(app_state.data_transformer).__name__
                if app_state.data_transformer is not None
                else "nenhum"
            )
            preview = ", ".join(feature_list[:10])
            if len(feature_list) > 10:
                preview += ", ..."
            logger.info(
                "Modelo '%s' carregado. Estrutura: %s. Transformador: %s. Features (%d): %s",
                model_file.name,
                app_state.model_structure,
                transformer_name,
                len(feature_list),
                preview or "não especificado",
            )
        except Exception as e:
            logger.error(f"Erro ao carregar o modelo de {model_file}: {e}", exc_info=True)
            app_state.model = None
            app_state.model_artifact = None
            app_state.data_transformer = None
            app_state.model_features = None
            app_state.model_structure = "unknown"
    else:
        logger.warning("Arquivo do modelo não encontrado em %s.", model_file)
        app_state.model = None
        app_state.model_artifact = None
        app_state.data_transformer = None
        app_state.model_features = None
        app_state.model_structure = "unknown"

# =============================================================================
# Ciclo de Vida da API (Lifespan)
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_model_and_setup_mlflow()
    yield
    logger.info("API encerrada.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://trustshield-dashboard:8501"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def ip_allowlist_middleware(request: Request, call_next):
    """Block requests from IPs that are not present in the allow list."""

    try:
        await enforce_ip_whitelist(request)
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return await call_next(request)

# =============================================================================
# DTOs (Data Transfer Objects)
# =============================================================================
class PredictRequest(BaseModel):
    """Contrato de entrada para o endpoint ``/predict``."""

    amount: float = Field(
        ..., gt=0, description="Valor da transação em dólares.", example=120.5
    )
    gender: str = Field(
        ..., description="Sexo do titular do cartão (Male/Female).", example="Male"
    )
    use_chip: str = Field(
        ...,
        description="Modo de utilização do cartão.",
        example="Swipe Transaction",
    )
    credit_score: int = Field(
        ..., ge=300, le=900, description="Pontuação de crédito do cliente.", example=720
    )
    num_credit_cards: int = Field(
        ..., ge=0, description="Número de cartões ativos do cliente.", example=3
    )
    per_capita_income: float = Field(
        ..., ge=0, description="Rendimento per capita anual.", example=23679
    )
    yearly_income: float = Field(
        ..., ge=0, description="Rendimento anual declarado.", example=48277
    )
    total_debt: float = Field(
        ..., ge=0, description="Dívida total estimada.", example=110153
    )

    class Config:
        extra = "forbid"

    @field_validator("gender")
    @classmethod
    def normalise_gender(cls, value: str) -> str:
        mapping = {
            "male": "Male",
            "m": "Male",
            "masculino": "Male",
            "female": "Female",
            "f": "Female",
            "feminino": "Female",
        }
        normalised = value.strip().lower()
        if normalised not in mapping:
            raise ValueError("gender deve ser 'Male' ou 'Female'.")
        return mapping[normalised]

    @field_validator("use_chip")
    @classmethod
    def normalise_use_chip(cls, value: str) -> str:
        allowed = {
            "swipe transaction": "Swipe Transaction",
            "chip transaction": "Chip Transaction",
            "online transaction": "Online Transaction",
            "contactless transaction": "Contactless Transaction",
        }
        normalised = value.strip().lower()
        if normalised not in allowed:
            raise ValueError(
                "use_chip deve ser Swipe Transaction, Chip Transaction, Online Transaction ou Contactless Transaction."
            )
        return allowed[normalised]

    def to_model_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        data.update(
            {
                "amount": float(data["amount"]),
                "per_capita_income": float(data["per_capita_income"]),
                "yearly_income": float(data["yearly_income"]),
                "total_debt": float(data["total_debt"]),
                "credit_score": int(data["credit_score"]),
                "num_credit_cards": int(data["num_credit_cards"]),
            }
        )
        return data


class PredictionOutput(BaseModel):
    """Estrutura da resposta da predição."""

    is_anomaly: bool = Field(
        ..., example=False, description="True indica possível anomalia."
    )
    score: float = Field(..., example=-0.2345)
    model_version: str

# =============================================================================
# Endpoints da API
# =============================================================================
from src.models.validation import ResilientTrustShieldValidator

@app.post("/validate", tags=["Validation"])
async def validate_model_endpoint(background_tasks: BackgroundTasks):
    """
    Dispara validação do modelo em background (drift, etc.)
    """
    logger.info("Requisição de validação recebida.")

    def run_validation_task():
        logger.info("Validação em background iniciada...")
        try:
            validator = ResilientTrustShieldValidator(config_path="config/config.yaml")
            validator.run_validation(
                data_path="data/features/featured_dataset.parquet",
                model_path="outputs/models/isolation_forest_optimized_29_20250815_054217.joblib",
                reference_data_path="data/features/featured_dataset.parquet",
                validation_types=["drift_detection"],
            )
            logger.info("Validação concluída com sucesso.")
        except Exception as e:
            logger.error(f"Erro na validação: {e}", exc_info=True)

    background_tasks.add_task(run_validation_task)
    return {"message": "Processo de validação do modelo iniciado em background."}

@app.get("/healthz", tags=["Health"])
def health_check():
    return {"status": "ok"}

@app.get("/readyz", tags=["Health"])
def ready_check():
    """Pronto para receber tráfego? (verifica modelo)"""
    if app_state.model is None:
        raise HTTPException(status_code=503, detail="Modelo não está disponível.")
    return {
        "model_loaded": True,
        "model_path": app_state.model_path,
        "mlflow_connected": app_state.mlflow_client is not None,
    }

@app.get("/status", tags=["Health"])
def get_status():
    """Status da API e do artefato carregado."""
    model_status = "OPERATIONAL" if app_state.model is not None else "UNAVAILABLE"
    model_name = Path(app_state.model_path).name if app_state.model is not None else "N/A"
    return {
        "status": model_status,
        "model_type": model_name,
        "mlflow_connected": app_state.mlflow_client is not None,
        "model_structure": app_state.model_structure,
    }

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(request: PredictRequest):
    """Executa a inferência com validação explícita e mensagens claras."""

    if app_state.model is None:
        raise HTTPException(status_code=503, detail="Modelo não está disponível.")

    try:
        payload = request.to_model_dict()
    except ValueError as exc:
        logger.warning("Entrada inválida em /predict: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    model_obj, transformer, model_features = _resolve_runtime_components(app_state)

    if model_obj is None:
        logger.error("Objeto de modelo não inicializado corretamente.")
        raise HTTPException(status_code=503, detail="Modelo não está disponível.")

    input_df = pd.DataFrame([payload])

    expected_columns: List[str] = []
    if model_features:
        expected_columns = list(model_features)
    elif transformer is not None and hasattr(transformer, "feature_names_in_"):
        expected_columns = list(getattr(transformer, "feature_names_in_"))

    if expected_columns:
        missing_columns = [col for col in expected_columns if col not in input_df.columns]
        if missing_columns:
            message = (
                "Campos obrigatórios em falta para predição: "
                + ", ".join(sorted(missing_columns))
            )
            logger.warning(message)
            raise HTTPException(status_code=422, detail=message)
        input_for_model = input_df.reindex(columns=expected_columns)
    else:
        input_for_model = input_df

    try:
        if transformer is not None:
            transformed_input = transformer.transform(input_for_model)
        else:
            numeric_only = input_for_model.select_dtypes(include=np.number)
            if numeric_only.empty:
                raise HTTPException(
                    status_code=422,
                    detail="Dados insuficientes: forneça campos numéricos para a predição.",
                )
            transformed_input = numeric_only
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Erro ao transformar dados de entrada: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=422,
            detail="Erro ao transformar dados para predição. Verifique os valores informados.",
        ) from exc

    try:
        score = None
        decision_fn = getattr(model_obj, "decision_function", None)
        if callable(decision_fn):
            score = float(decision_fn(transformed_input)[0])
        else:
            score_samples_fn = getattr(model_obj, "score_samples", None)
            if callable(score_samples_fn):
                score = float(score_samples_fn(transformed_input)[0])

        raw_prediction = model_obj.predict(transformed_input)[0]
    except Exception as exc:
        logger.error("Erro durante a predição: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail="Erro interno, consultar logs."
        ) from exc

    is_anomaly = bool(raw_prediction == -1 or raw_prediction is True)
    response = {
        "is_anomaly": is_anomaly,
        "score": float(score) if score is not None else 0.0,
        "model_version": Path(app_state.model_path).name,
    }
    logger.info(
        "Predição concluída | anomalia=%s | score=%.4f | modelo=%s",
        is_anomaly,
        response["score"],
        response["model_version"],
    )
    return response

@app.post("/reload", tags=["Admin"])
async def reload_model(background_tasks: BackgroundTasks):
    """Recarrega modelo e reconfigura MLflow em background."""
    logger.info("Requisição para recarregar o modelo recebida.")
    background_tasks.add_task(load_model_and_setup_mlflow)
    return {"message": "Processo de recarga do modelo iniciado."}
