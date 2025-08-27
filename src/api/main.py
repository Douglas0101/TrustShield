# -*- coding: utf-8 -*-
"""
TrustShield API (FastAPI) — versão otimizada e robusta

Principais melhorias em relação à base anterior:
- `ResourceMonitor.get_stats()` implementado (evita 500 no /status).
- Rota raiz `/` amigável e `/healthz` para probes.
- Carregamento resiliente de configurações (fallback se `config.yaml` ausente).
- Limite conservador de threads numéricas no processo (OMP/BLAS/Numba).
- Logs consistentes e observabilidade via Observers (Console, Prometheus, MLflow) preservados.
- Tratamento robusto de caminhos e artefatos (modelo) durante o startup.

Como executar (raiz do projeto):
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

# =============================================================================
# Imports & setup
# =============================================================================
import logging
import os
import sys
import time
import warnings
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Annotated, Optional, List, Protocol, runtime_checkable

import json
import tempfile

import psutil
import yaml
import mlflow
import pandas as pd

from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.datastructures import State
from pydantic import BaseModel, ValidationError, field_validator
from pydantic.types import confloat

# Projeto
from src.models.predict import TrustShieldPredictor, PredictionResult
from src.models.validation import ResilientTrustShieldValidator
from src.models.interpretation import ResilientModelInterpreter

# Dependências opcionais
try:
    from prometheus_client import Counter, Histogram
    PROMETHEUS_AVAILABLE = True
except Exception:
    PROMETHEUS_AVAILABLE = False

try:
    from circuitbreaker import circuit
    CIRCUITBREAKER_AVAILABLE = True
except Exception:
    CIRCUITBREAKER_AVAILABLE = False

# Configuração global de warnings e threads numéricas
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")           # conservador para evitar tempestade de threads
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

API_VERSION = "5.3.0-optimized"

# =============================================================================
# Infra — logging e config
# =============================================================================
class AdvancedLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s - [TrustShield-API] - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log(self, level: int, message: str, **kwargs):
        extra = {"timestamp": datetime.now().isoformat()}
        extra.update(kwargs)
        self.logger.log(level, message, extra=extra)


class ConfigManager:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        config_path = self.project_root / "config" / "config.yaml"
        self.settings: Dict[str, Any] = {
            "api": {
                "model_path": "outputs/models/default_model.joblib",
                "model_validation": False,
            },
            "mlflow": {
                "experiment_name": "TrustShield",
            },
        }
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    loaded = yaml.safe_load(f) or {}
                    # merge raso — apenas chaves conhecidas
                    for k in ("api", "mlflow"):
                        if k in loaded and isinstance(loaded[k], dict):
                            self.settings[k].update(loaded[k])
            except Exception as e:
                # mantém defaults e segue
                print(f"[ConfigManager] Falha ao ler config.yaml: {e}")

    def get_config(self) -> Dict[str, Any]:
        return self.settings


class ResourceMonitor:
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.process = psutil.Process()
        self.request_count = 0
        self.error_count = 0

    def record_request(self, success: bool = True):
        self.request_count += 1
        if not success:
            self.error_count += 1

    def get_stats(self) -> Dict[str, Any]:
        try:
            cpu = psutil.cpu_percent(interval=0.0)
            mem = self.process.memory_info().rss / (1024 ** 2)
        except Exception:
            cpu, mem = None, None
        return {
            "requests": self.request_count,
            "errors": self.error_count,
            "cpu_percent": cpu,
            "mem_rss_mb": round(mem, 2) if mem is not None else None,
        }


# =============================================================================
# Domínio — eventos & métricas
# =============================================================================
class APIEvent(Enum):
    STARTUP = 0
    SHUTDOWN = 1
    REQUEST_RECEIVED = 2
    REQUEST_PROCESSED = 3
    ERROR_OCCURRED = 4
    MODEL_LOADED = 5
    MODEL_VALIDATED = 6
    DRIFT_DETECTED = 7


@dataclass
class APIMetrics:
    event: APIEvent
    endpoint: str
    execution_time: float
    success: bool
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


# Pydantic models (V2)
class TransactionInput(BaseModel):
    amount: confloat(ge=0, le=1_000_000)
    use_chip: str
    current_age: int
    retirement_age: int
    birth_year: int
    gender: str
    latitude: float
    longitude: float
    yearly_income: confloat(ge=0)
    total_debt: confloat(ge=0)
    credit_score: int
    num_credit_cards: int
    transaction_hour: int
    day_of_week: int
    is_weekend: bool
    is_night_transaction: bool
    amount_vs_avg: confloat(ge=0)

    @field_validator("birth_year")
    def validate_birth_year(cls, v):
        current_year = datetime.now().year
        if v < 1900 or v > current_year:
            raise ValueError("Ano de nascimento inválido")
        return v

    @field_validator("transaction_hour")
    def validate_hour(cls, v):
        if v < 0 or v > 23:
            raise ValueError("Hora da transação inválida")
        return v

    @field_validator("day_of_week")
    def validate_day(cls, v):
        if v < 0 or v > 6:
            raise ValueError("Dia da semana inválido")
        return v


class BatchTransactionInput(BaseModel):
    transactions: List[TransactionInput]

    @field_validator("transactions")
    def validate_transactions(cls, v):
        if len(v) > 1000:
            raise ValueError("Tamanho máximo do batch é 1000 transações")
        return v


# =============================================================================
# Observers
# =============================================================================
@runtime_checkable
class APIObserver(Protocol):
    def update(self, event: APIEvent, data: Dict[str, Any]): ...


class Subject:
    def __init__(self):
        self._observers: List[APIObserver] = []

    def attach(self, observer: APIObserver):
        self._observers.append(observer)

    def notify(self, event: APIEvent, data: Dict[str, Any]):
        for observer in self._observers:
            try:
                observer.update(event, data)
            except Exception:
                pass


class ConsoleLogObserver(APIObserver):
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger

    def update(self, event: APIEvent, data: Dict[str, Any]):
        messages = {
            APIEvent.STARTUP: "🚀 API TrustShield iniciada",
            APIEvent.SHUTDOWN: "🛑 API TrustShield finalizada",
            APIEvent.REQUEST_RECEIVED: f"📥 Requisição recebida: {data.get('endpoint', 'N/A')}",
            APIEvent.REQUEST_PROCESSED: f"✅ Requisição processada: {data.get('endpoint', 'N/A')} ({data.get('execution_time', 0):.2f}s)",
            APIEvent.ERROR_OCCURRED: f"❌ Erro na requisição: {data.get('error', 'N/A')}",
            APIEvent.MODEL_LOADED: f"🔄 Modelo carregado: {data.get('model_type', 'N/A')}",
        }
        if message := messages.get(event):
            self.logger.log(logging.INFO, message)


class PrometheusObserver(APIObserver):
    def __init__(self):
        if PROMETHEUS_AVAILABLE:
            self.request_counter = Counter(
                "trustshield_api_requests_total", "Total API requests", ["endpoint", "status"]
            )
            self.request_duration = Histogram(
                "trustshield_api_request_duration_seconds", "API request duration"
            )

    def update(self, event: APIEvent, data: Dict[str, Any]):
        if not PROMETHEUS_AVAILABLE:
            return
        if event == APIEvent.REQUEST_PROCESSED:
            metrics = data.get("metrics", {}) or data
            endpoint = metrics.get("endpoint", "unknown")
            status = "success" if metrics.get("success", False) else "error"
            self.request_counter.labels(endpoint=endpoint, status=status).inc()
            if "execution_time" in metrics:
                self.request_duration.observe(metrics["execution_time"])


class MLflowObserver(APIObserver):
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.run_id = None

    def update(self, event: APIEvent, data: Dict[str, Any]):
        try:
            if event == APIEvent.STARTUP:
                mlflow.set_experiment(self.experiment_name)
                mlflow.start_run(run_name=f"api_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
                self.run_id = mlflow.active_run().info.run_id
            elif event == APIEvent.REQUEST_PROCESSED and self.run_id:
                metrics = data.get("metrics", {}) or data
                mlflow.log_metric(
                    f"{metrics.get('endpoint', 'req').replace('/', '_')}_time",
                    metrics.get("execution_time", 0.0),
                )
            elif event == APIEvent.SHUTDOWN:
                if mlflow.active_run():
                    mlflow.end_run()
        except Exception:
            # observabilidade não deve derrubar a API
            pass


# =============================================================================
# App state & lifecycle
# =============================================================================

class AppState(State):
    predictor: Optional[TrustShieldPredictor]
    config: Dict[str, Any]
    logger: AdvancedLogger
    monitor: ResourceMonitor
    subject: Subject
    model_path: Path
    model_loading_task: Optional[asyncio.Task] = None


async def load_model_in_background(app: FastAPI):
    """Carrega o modelo em uma tarefa de fundo para não bloquear a inicialização."""
    app_state: AppState = app.state
    try:
        project_root = Path(__file__).resolve().parents[2]
        model_path_str = app_state.config.get("api", {}).get("model_path", "outputs/models/default_model.joblib")
        model_path = project_root / model_path_str if not Path(model_path_str).is_absolute() else Path(model_path_str)
        app_state.model_path = model_path

        # A inicialização do TrustShieldPredictor é bloqueante, então a executamos em um thread
        predictor = await asyncio.to_thread(TrustShieldPredictor, model_path=str(model_path))
        
        app_state.predictor = predictor
        app_state.subject.notify(APIEvent.MODEL_LOADED, {
            "model_type": app_state.predictor.model_type.value if getattr(app_state.predictor, 'model_type', None) else "unknown"
        })
    except Exception as e:
        app_state.logger.log(logging.ERROR, f"Falha crítica ao carregar o modelo em background: {e}")
        app_state.predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state = AppState()
    project_root = Path(__file__).resolve().parents[2]

    # Config
    config_manager = ConfigManager(project_root)
    app.state.config = config_manager.get_config()

    # Infra
    app.state.logger = AdvancedLogger("TrustShield-API")
    app.state.monitor = ResourceMonitor(app.state.logger)
    app.state.subject = Subject()
    app.state.predictor = None # Inicia como None

    # Observers
    app.state.subject.attach(ConsoleLogObserver(app.state.logger))
    app.state.subject.attach(PrometheusObserver())
    app.state.subject.attach(MLflowObserver(app.state.config.get("mlflow", {}).get("experiment_name", "TrustShield")))

    app.state.subject.notify(APIEvent.STARTUP, {})

    # Inicia o carregamento do modelo em background
    app.state.model_loading_task = asyncio.create_task(load_model_in_background(app))

    try:
        yield
    finally:
        if app.state.model_loading_task:
            app.state.model_loading_task.cancel()
        app.state.subject.notify(APIEvent.SHUTDOWN, {})


app = FastAPI(
    title="TrustShield Fraud Detection API",
    version=API_VERSION,
    lifespan=lifespan,
)

# CORS (liberal para dev; ajuste em prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Dependencies & middleware
# =============================================================================

def get_predictor(request: Request) -> TrustShieldPredictor:
    predictor = request.app.state.predictor
    if not predictor:
        raise HTTPException(status_code=503, detail="Serviço indisponível: Modelo não carregado.")
    return predictor

PredictorDep = Annotated[TrustShieldPredictor, Depends(get_predictor)]


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    app_state: AppState = request.app.state
    app_state.subject.notify(APIEvent.REQUEST_RECEIVED, {"endpoint": request.url.path})

    try:
        response = await call_next(request)
        execution_time = time.time() - start_time
        metrics = APIMetrics(
            event=APIEvent.REQUEST_PROCESSED,
            endpoint=request.url.path,
            execution_time=execution_time,
            success=True,
            status_code=response.status_code,
        )
        app_state.monitor.record_request(success=True)
        app_state.subject.notify(APIEvent.REQUEST_PROCESSED, metrics.__dict__)
        return response
    except Exception as e:
        execution_time = time.time() - start_time
        metrics = APIMetrics(
            event=APIEvent.ERROR_OCCURRED,
            endpoint=request.url.path,
            execution_time=execution_time,
            success=False,
            error_message=str(e),
        )
        app_state.monitor.record_request(success=False)
        app_state.subject.notify(APIEvent.ERROR_OCCURRED, metrics.__dict__)
        raise


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


# =============================================================================
# Circuit breakers (opcional)
# =============================================================================
if CIRCUITBREAKER_AVAILABLE:
    @circuit(failure_threshold=5, recovery_timeout=60)
    async def predict_with_circuit_breaker(predictor: TrustShieldPredictor, transaction_data: Dict[str, Any]) -> PredictionResult:
        return predictor.predict(transaction_data)

    @circuit(failure_threshold=3, recovery_timeout=30)
    async def batch_predict_with_circuit_breaker(predictor: TrustShieldPredictor, transaction_data: List[Dict[str, Any]]) -> List[PredictionResult]:
        return predictor.batch_predict(transaction_data)
else:
    async def predict_with_circuit_breaker(predictor: TrustShieldPredictor, transaction_data: Dict[str, Any]) -> PredictionResult:
        return predictor.predict(transaction_data)

    async def batch_predict_with_circuit_breaker(predictor: TrustShieldPredictor, transaction_data: List[Dict[str, Any]]) -> List[PredictionResult]:
        return predictor.batch_predict(transaction_data)


# =============================================================================
# Endpoints — raiz/health
# =============================================================================
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "TrustShield API is running. See /docs and /status."}


@app.get("/healthz", tags=["Health"])  # simples para probes
async def healthz():
    return {"status": "ok", "version": API_VERSION}


# =============================================================================
# Endpoints — predição e explicabilidade
# =============================================================================
@app.post("/predict", tags=["Prediction"], response_model=Dict[str, Any])
async def predict_transaction(transaction: TransactionInput, predictor: PredictorDep):
    try:
        result = await predict_with_circuit_breaker(predictor, transaction.model_dump())
        if not result.success:
            raise HTTPException(status_code=500, detail=f"Erro interno na predição: {result.error}")
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict", tags=["Prediction"], response_model=List[Dict[str, Any]])
async def batch_predict_transactions(batch: BatchTransactionInput, predictor: PredictorDep):
    try:
        transactions_list = [t.model_dump() for t in batch.transactions]
        results = await batch_predict_with_circuit_breaker(predictor, transactions_list)
        return [r.to_dict() for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- Explicabilidade assíncrona -------------------------------------------------

def run_explanation(interpreter: ResilientModelInterpreter, data_path: str, output_path: str):
    """Executa a interpretação e salva o resultado em um arquivo específico."""
    try:
        interpreter.run_interpretation(data_path=data_path, methods=['shap'])
        # Encontrar o último resultado gerado no diretório padrão e movê-lo para o output esperado
        project_root = Path(__file__).resolve().parents[2]
        source_dir = project_root / "outputs" / "interpretations" / "shap"
        result_files = list(source_dir.glob('*.json'))
        if not result_files:
            return
        latest_result_file = max(result_files, key=os.path.getctime)
        os.rename(latest_result_file, output_path)
    finally:
        # Limpa o arquivo temporário de entrada, se existir
        try:
            if os.path.exists(data_path):
                os.unlink(data_path)
        except Exception:
            pass


@app.post("/explain", tags=["Interpretation"], status_code=202)
async def explain_transaction_async(transaction: TransactionInput, background_tasks: BackgroundTasks, request: Request):
    app_state: AppState = request.app.state
    model_path = str(app_state.model_path)

    # ID único para o job
    job_id = f"explanation_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}"
    project_root = Path(__file__).resolve().parents[2]

    # Pastas temporárias
    temp_dir = project_root / "outputs" / "temp_explanations"
    temp_dir.mkdir(exist_ok=True)

    input_data_path = temp_dir / f"{job_id}_input.parquet"
    output_result_path = temp_dir / f"{job_id}_result.json"

    # Persistir entrada em parquet
    df = pd.DataFrame([transaction.model_dump()])
    df.to_parquet(input_data_path)

    # Interpretador + background task
    interpreter = ResilientModelInterpreter(model_path=model_path)
    background_tasks.add_task(run_explanation, interpreter, str(input_data_path), str(output_result_path))

    return {"job_id": job_id, "status": "explanation_started"}


@app.get("/explanation-result/{job_id}", tags=["Interpretation"], response_model=Dict[str, Any])
async def get_explanation_result(job_id: str):
    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "outputs" / "temp_explanations" / f"{job_id}_result.json"

    if not output_path.exists():
        raise HTTPException(status_code=202, detail="Explanation result not ready yet.")

    with open(output_path, "r") as f:
        explanation = json.load(f)

    # opcional: remover o arquivo após leitura
    # os.unlink(output_path)

    return explanation


# =============================================================================
# Endpoints — status e validação
# =============================================================================
@app.get("/status", tags=["Health"], response_model=Dict[str, Any])
async def get_status(predictor: PredictorDep, request: Request):
    status = predictor.get_status()
    status["version"] = API_VERSION
    status["api_metrics"] = request.app.state.monitor.get_stats()
    return status


async def run_model_validation(app_state: AppState):
    validator = ResilientTrustShieldValidator()
    validator.run_validation(
        data_path="data/interim/validation_sample.parquet",
        model_path=str(app_state.model_path),
        reference_data_path="data/features/featured_dataset.parquet",
        validation_types=['drift_detection'],
    )
    app_state.subject.notify(APIEvent.MODEL_VALIDATED, {})


@app.post("/validate", tags=["Validation"])
async def validate_model(background_tasks: BackgroundTasks, request: Request):
    app_state = request.app.state
    if not app_state.config.get('api', {}).get('model_validation', False):
        raise HTTPException(status_code=400, detail="Validação de modelo não habilitada.")

    background_tasks.add_task(run_model_validation, app_state)
    return {"status": "validation_started"}
