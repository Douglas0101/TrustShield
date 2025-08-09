# src/api/main.py

"""
API Principal do Projeto TrustShield - Versão Otimizada e Corrigida
Versão: 5.2.0 - Enterprise MLOps Integration (Linter-Clean & Enhanced Usability)

Melhorias Implementadas:
✅ Código alinhado com Pydantic V2 (@field_validator, .model_dump()).
✅ Resolução de todas as referências de importação e atributos.
✅ Correção de erros de tipo em chamadas de Enum.
✅ Implementação do padrão Observer para monitoramento modular (MLflow, Prometheus).
✅ Uso de 'time' e 'dataclasses' para métricas de performance estruturadas.
✅ Integração profunda com MLflow para observabilidade da API.

Autor: TrustShield Team & IA Gemini
Versão: 5.2.0-enterprise-mlops
Data: 2025-08-12
"""

# =====================================================================================
# 📦 IMPORTS E CONFIGURAÇÕES INICIAIS
# =====================================================================================

import logging
import os
import sys
import time
import warnings
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Annotated, Optional, List, Protocol, runtime_checkable

import yaml
import psutil
import mlflow
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from starlette.datastructures import State
from pydantic import BaseModel, ValidationError, field_validator
from pydantic.types import confloat

# Importações do projeto
from src.models.predict import TrustShieldPredictor, PredictionResult
from src.models.validation import ResilientTrustShieldValidator

# Dependências opcionais de Engenharia de IA
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    from dynaconf import Dynaconf

    DYNACONF_AVAILABLE = True
except ImportError:
    DYNACONF_AVAILABLE = False

try:
    from circuitbreaker import circuit

    CIRCUITBREAKER_AVAILABLE = True
except ImportError:
    CIRCUITBREAKER_AVAILABLE = False

# Configurações globais
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# =====================================================================================
# 🏗️ CAMADA DE INFRAESTRUTURA - SERVIÇOS DE SUPORTE
# =====================================================================================

class AdvancedLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - [TrustShield-API] - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log(self, level: int, message: str, **kwargs):
        extra = {'timestamp': datetime.now().isoformat()}
        extra.update(kwargs)
        self.logger.log(level, message, extra=extra)


class ConfigManager:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        config_path = self.project_root / "config" / "config.yaml"
        with open(config_path, 'r') as f:
            self.settings = yaml.safe_load(f)

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


# =====================================================================================
# 🏗️ CAMADA DE DOMÍNIO - LÓGICA DE NEGÓCIO CENTRAL
# =====================================================================================

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


# Modelos Pydantic para validação de entrada (Pydantic V2)
class TransactionInput(BaseModel):
    amount: confloat(ge=0, le=1000000)
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

    @field_validator('birth_year')
    def validate_birth_year(cls, v):
        current_year = datetime.now().year
        if v < 1900 or v > current_year:
            raise ValueError('Ano de nascimento inválido')
        return v

    @field_validator('transaction_hour')
    def validate_hour(cls, v):
        if v < 0 or v > 23:
            raise ValueError('Hora da transação inválida')
        return v

    @field_validator('day_of_week')
    def validate_day(cls, v):
        if v < 0 or v > 6:
            raise ValueError('Dia da semana inválido')
        return v


class BatchTransactionInput(BaseModel):
    transactions: List[TransactionInput]

    @field_validator('transactions')
    def validate_transactions(cls, v):
        if len(v) > 1000:
            raise ValueError('Tamanho máximo do batch é 1000 transações')
        return v


# =====================================================================================
# 🔧 CAMADA DE APLICAÇÃO - CASOS DE USO E ORQUESTRAÇÃO
# =====================================================================================

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
            observer.update(event, data)


# =====================================================================================
# 🏭 CAMADA DE INFRAESTRUTURA - IMPLEMENTAÇÕES CONCRETAS
# =====================================================================================

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
            self.request_counter = Counter('trustshield_api_requests_total', 'Total API requests',
                                           ['endpoint', 'status'])
            self.request_duration = Histogram('trustshield_api_request_duration_seconds', 'API request duration')

    def update(self, event: APIEvent, data: Dict[str, Any]):
        if not PROMETHEUS_AVAILABLE:
            return
        if event == APIEvent.REQUEST_PROCESSED:
            metrics = data.get('metrics', {})
            endpoint = metrics.get('endpoint', 'unknown')
            status = 'success' if metrics.get('success', False) else 'error'
            self.request_counter.labels(endpoint=endpoint, status=status).inc()
            if 'execution_time' in metrics:
                self.request_duration.observe(metrics['execution_time'])


class MLflowObserver(APIObserver):
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.run_id = None

    def update(self, event: APIEvent, data: Dict[str, Any]):
        if event == APIEvent.STARTUP:
            mlflow.start_run(run_name=f"api_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            self.run_id = mlflow.active_run().info.run_id
        elif event == APIEvent.REQUEST_PROCESSED:
            if self.run_id:
                metrics = data.get('metrics', {})
                mlflow.log_metric(f"{metrics.get('endpoint', 'req').replace('/', '_')}_time",
                                  metrics.get('execution_time', 0))
        elif event == APIEvent.SHUTDOWN:
            if mlflow.active_run():
                mlflow.end_run()


# =====================================================================================
# 🎼 ORQUESTRADOR - O SERVIÇO PRINCIPAL DA APLICAÇÃO
# =====================================================================================

class AppState(State):
    predictor: Optional[TrustShieldPredictor]
    config: Dict[str, Any]
    logger: AdvancedLogger
    monitor: ResourceMonitor
    subject: Subject
    model_path: Path


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state = AppState()
    project_root = Path(__file__).resolve().parents[2]
    config_manager = ConfigManager(project_root)
    app.state.config = config_manager.get_config()
    app.state.logger = AdvancedLogger('TrustShield-API')
    app.state.monitor = ResourceMonitor(app.state.logger)
    app.state.subject = Subject()

    app.state.subject.attach(ConsoleLogObserver(app.state.logger))
    app.state.subject.attach(PrometheusObserver())
    app.state.subject.attach(MLflowObserver(app.state.config.get('mlflow', {}).get('experiment_name', 'TrustShield')))

    app.state.subject.notify(APIEvent.STARTUP, {})

    try:
        model_path_str = app.state.config.get('api', {}).get('model_path', 'outputs/models/default_model.joblib')
        model_path = project_root / model_path_str if not Path(model_path_str).is_absolute() else Path(model_path_str)
        app.state.model_path = model_path

        app.state.predictor = TrustShieldPredictor(model_path=str(model_path))

        app.state.subject.notify(APIEvent.MODEL_LOADED, {
            "model_type": app.state.predictor.model_type.value if app.state.predictor.model_type else "unknown"
        })

        if PROMETHEUS_AVAILABLE:
            start_http_server(8001)

        yield
    finally:
        app.state.subject.notify(APIEvent.SHUTDOWN, {})


app = FastAPI(
    title="TrustShield Fraud Detection API",
    version="5.2.0-linter-clean",
    lifespan=lifespan
)


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
            status_code=response.status_code
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
            error_message=str(e)
        )
        app_state.monitor.record_request(success=False)
        app_state.subject.notify(APIEvent.ERROR_OCCURRED, metrics.__dict__)
        raise


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


if CIRCUITBREAKER_AVAILABLE:
    @circuit(failure_threshold=5, recovery_timeout=60)
    async def predict_with_circuit_breaker(predictor: TrustShieldPredictor,
                                           transaction_data: Dict[str, Any]) -> PredictionResult:
        return predictor.predict(transaction_data)


    @circuit(failure_threshold=3, recovery_timeout=30)
    async def batch_predict_with_circuit_breaker(predictor: TrustShieldPredictor,
                                                 transaction_data: List[Dict[str, Any]]) -> List[PredictionResult]:
        return predictor.batch_predict(transaction_data)
else:
    async def predict_with_circuit_breaker(predictor: TrustShieldPredictor,
                                           transaction_data: Dict[str, Any]) -> PredictionResult:
        return predictor.predict(transaction_data)


    async def batch_predict_with_circuit_breaker(predictor: TrustShieldPredictor,
                                                 transaction_data: List[Dict[str, Any]]) -> List[PredictionResult]:
        return predictor.batch_predict(transaction_data)


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


@app.get("/status", tags=["Health Check"], response_model=Dict[str, Any])
async def get_status(predictor: PredictorDep, request: Request):
    status = predictor.get_status()
    status['api_metrics'] = request.app.state.monitor.get_stats()
    return status


async def run_model_validation(app_state: AppState):
    validator = ResilientTrustShieldValidator()
    validator.run_validation(
        data_path="data/interim/validation_sample.parquet",
        model_path=str(app_state.model_path),
        validation_types=['model_performance']
    )
    app_state.subject.notify(APIEvent.MODEL_VALIDATED, {})


@app.post("/validate", tags=["Validation"])
async def validate_model(background_tasks: BackgroundTasks, request: Request):
    app_state = request.app.state
    if not app_state.config.get('api', {}).get('model_validation', False):
        raise HTTPException(status_code=400, detail="Validação de modelo não habilitada.")

    background_tasks.add_task(run_model_validation, app_state)
    return {"status": "validation_started"}