# -*- coding: utf-8 -*-
"""
Módulo de Otimização de Hiperparâmetros do Projeto TrustShield
Versão: 4.3.0 - Engenharia Definitiva

Melhorias Implementadas:
✅ Código 100% calibrado e livre de warnings de linter.
✅ Supressão inteligente de falsos positivos de "import não utilizado" com '# noqa: F401',
   preservando a robustez da arquitetura dinâmica (Strategy Pattern).
✅ Manutenção de todos os imports críticos, incluindo 'optuna'.
✅ Arquitetura de camadas lógicas mantida para máxima clareza e organização.

Autor: TrustShield Team & IA Gemini
Versão: 4.3.0-definitive-engineered
Data: 2025-08-13
"""

# =====================================================================================
# 📦 IMPORTS E CONFIGURAÇÕES INICIAIS
# =====================================================================================

import argparse
import logging
import os
import psutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Protocol, runtime_checkable
from enum import Enum
from dataclasses import dataclass, field, asdict

# Imports essenciais para o funcionamento dinâmico do módulo.
# O comentário '# noqa: F401' instrui o linter a ignorar o falso positivo de "import não utilizado",
# pois estas bibliotecas são usadas dinamicamente pelas classes de Estratégia e Observer.
import joblib  # noqa: F401
try:  # pragma: no cover - dependência opcional em ambientes de CI
    import mlflow  # type: ignore
except ImportError:  # pragma: no cover - fallback seguro
    mlflow = None  # type: ignore
import numpy as np  # noqa: F401
try:  # pragma: no cover - optuna é opcional para alguns fluxos
    import optuna  # type: ignore
    OPTUNA_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback seguro
    optuna = None  # type: ignore
    OPTUNA_AVAILABLE = False
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest  # noqa: F401
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # noqa: F401

# Tratamento robusto de dependências opcionais
try:
    import great_expectations as ge
    from great_expectations.core import ExpectationSuite, ExpectationConfiguration
    from great_expectations.checkpoint import Checkpoint

    GE_AVAILABLE = True
except ImportError:
    ge = ExpectationSuite = ExpectationConfiguration = None
    GE_AVAILABLE = False

try:
    from dynaconf import Dynaconf

    DYNACONF_AVAILABLE = True
except ImportError:
    Dynaconf = None
    DYNACONF_AVAILABLE = False

try:
    from circuitbreaker import circuit

    CIRCUITBREAKER_AVAILABLE = True
except ImportError:

    def circuit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    CIRCUITBREAKER_AVAILABLE = False

try:
    import ray
    from ray.tune.schedulers import AsyncHyperBandScheduler
    from ray.tune.search.optuna import OptunaSearch

    RAY_AVAILABLE = True
except ImportError:
    ray = AsyncHyperBandScheduler = OptunaSearch = None
    RAY_AVAILABLE = False

# Configurações globais
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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
                "%(asctime)s - [TrustShield-Optimizer] - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log(self, level: int, message: str, **kwargs):
        reserved_keys = {"exc_info", "stack_info", "stacklevel"}
        log_kwargs: Dict[str, Any] = {}
        for key in reserved_keys:
            if key in kwargs:
                log_kwargs[key] = kwargs.pop(key)

        extra_payload = kwargs.pop("extra", {}) or {}
        if not isinstance(extra_payload, dict):
            extra_payload = dict(extra_payload)

        extra_payload = {"timestamp": datetime.now().isoformat(), **extra_payload, **kwargs}
        log_kwargs["extra"] = extra_payload

        self.logger.log(level, message, **log_kwargs)


class ConfigManager:
    def __init__(self, project_root: Path, logger: AdvancedLogger):
        self.project_root = project_root
        self.logger = logger
        if DYNACONF_AVAILABLE and Dynaconf:
            self.settings = Dynaconf(
                settings_files=[str(project_root / "config" / "config.yaml")],
                environments=False,
                env_switcher="ENV_FOR_DYNACONF",
                load_dotenv=True,
            )
        else:
            self.settings = None

    def get_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        if DYNACONF_AVAILABLE and self.settings:
            config = self._load_from_dynaconf()

        if config and not self._is_valid_config(config):
            self.logger.log(
                logging.WARNING,
                "Configuração via Dynaconf incompleta - aplicando fallback para YAML.",
            )

        if not self._is_valid_config(config):
            config = self._load_from_yaml()

        self._apply_environment_overrides(config)
        return config

    def _load_from_dynaconf(self) -> Dict[str, Any]:
        try:
            raw_config = self.settings.as_dict()  # type: ignore[union-attr]
            config = self._normalize_keys(raw_config)
            config = self.settings.as_dict()  # type: ignore[union-attr]
            self.logger.log(
                logging.INFO,
                f"Config loaded from dynaconf with keys: {list(config.keys())}",
            )
            return config
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.log(
                logging.WARNING,
                f"Falha ao carregar configuração via Dynaconf: {exc}",
                exc_info=True,
            )
            return {}

    def _load_from_yaml(self) -> Dict[str, Any]:
        config_path = self.project_root / "config" / "config.yaml"
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file) or {}
        self.logger.log(
            logging.INFO, f"Config loaded from YAML: {config_path.relative_to(self.project_root)}"
        )
        return config

    @staticmethod
    def _is_valid_config(config: Dict[str, Any]) -> bool:
        return bool(
            isinstance(config, dict)
            and isinstance(config.get("hyper_optimization"), dict)
            and config["hyper_optimization"].get("space")
        )

    def _apply_environment_overrides(self, config: Dict[str, Any]):
        if "hyper_optimization" not in config:
            config["hyper_optimization"] = {}

        env = os.getenv("ENV", "development").lower()
        hyper_cfg = config["hyper_optimization"]

        if env == "production":
            hyper_cfg["n_trials"] = 100
            hyper_cfg["early_stopping"] = True
        elif env == "staging":
            hyper_cfg["n_trials"] = max(int(hyper_cfg.get("n_trials", 50)), 50)
            
    def _normalize_keys(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for key, value in payload.items():
            normalized_key = key.lower() if isinstance(key, str) else key
            if isinstance(value, dict):
                normalized[normalized_key] = self._normalize_keys(value)
            else:
                normalized[normalized_key] = value
        return normalized

class ResourceMonitor:
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.process = psutil.Process()
        self.start_time = time.time()
        self.peak_memory = 0
        self.trial_times: List[float] = []
        self.memory_history: List[float] = []

    def update_peak_memory(self):
        current_memory = self.process.memory_info().rss / (1024**3)
        self.memory_history.append(current_memory)
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory

    def record_trial_time(self, trial_time: float):
        self.trial_times.append(trial_time)

    @property
    def average_trial_time(self) -> float:
        return float(np.mean(self.trial_times)) if self.trial_times else 0.0

    @property
    def max_trial_time(self) -> float:
        return float(max(self.trial_times)) if self.trial_times else 0.0

    @property
    def latest_memory_gb(self) -> float:
        return self.memory_history[-1] if self.memory_history else 0.0

    def reset(self):
        self.start_time = time.time()
        self.peak_memory = 0
        self.trial_times.clear()
        self.memory_history.clear()


class DataValidator:
    def __init__(self, config: Dict[str, Any], logger: AdvancedLogger):
        self.config = config
        self.logger = logger
        if GE_AVAILABLE and ge:
            self.context = ge.get_context()
        else:
            self.context = None

    def validate(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        if not self.context:
            self.logger.log(
                logging.WARNING, "Great Expectations não disponível - pulando validação"
            )
            return True, {}
        try:
            datasource = self.context.sources.add_pandas(name="my_pandas_datasource")
            data_asset = datasource.add_dataframe_asset(
                name="my_dataframe_asset", dataframe=df
            )

            expectation_suite_name = "optimization_suite"
            self.context.add_or_update_expectation_suite(
                expectation_suite_name=expectation_suite_name
            )

            validator = self.context.get_validator(
                batch_request=data_asset.build_batch_request(),
                expectation_suite_name=expectation_suite_name,
            )

            validator.expect_column_values_to_not_be_null("amount")
            validator.expect_table_row_count_to_be_between(
                min_value=100, max_value=15000000
            )
            validator.save_expectation_suite(discard_failed_expectations=False)

            checkpoint = Checkpoint(
                name="my_in_memory_checkpoint",
                data_context=self.context,
                validations=[
                    {
                        "batch_request": data_asset.build_batch_request(),
                        "expectation_suite_name": expectation_suite_name,
                    },
                ],
                action_list=[
                    {
                        "name": "store_validation_result",
                        "action": {"class_name": "StoreValidationResultAction"},
                    },
                    {
                        "name": "update_data_docs",
                        "action": {"class_name": "UpdateDataDocsAction"},
                    },
                ],
            )
            checkpoint_result = checkpoint.run()

            run_results = checkpoint_result.run_results
            validation_result_identifier = list(run_results.keys())[0]
            validation_result = run_results[validation_result_identifier][
                "validation_result"
            ]

            validation_results = {
                "success": validation_result.success,
                "statistics": validation_result.statistics,
                "failed_expectations": len(
                    [r for r in validation_result.results if not r.success]
                ),
            }

            if not validation_results["success"]:
                self.logger.log(
                    logging.ERROR, f"Validation failed. Results: {validation_result}"
                )

            self.logger.log(
                logging.INFO if validation_results["success"] else logging.ERROR,
                f"Validação de dados: {'Sucesso' if validation_results['success'] else 'Falha'}",
                **validation_results,
            )
            return validation_results["success"], validation_results
        except Exception as e:
            self.logger.log(logging.ERROR, f"Erro na validação de dados: {e}")
            return False, {"error": str(e)}


# =====================================================================================
# 🏗️ CAMADA DE DOMÍNIO - LÓGICA DE NEGÓCIO CENTRAL
# =====================================================================================


class OptimizationMethod(Enum):
    OPTUNA = "optuna"
    RAY_TUNE = "ray_tune"


@dataclass
class OptimizationMetrics:
    method: OptimizationMethod
    total_time: float
    best_score: float
    n_trials: int
    n_successful_trials: int
    peak_memory_gb: float
    cpu_usage_percent: float = 0.0
    avg_trial_time: float = 0.0
    max_trial_time: float = 0.0
    latest_memory_gb: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["method"] = self.method.value
        data["timestamp"] = self.timestamp.isoformat()
        return data

    def to_numeric_metrics(self) -> Dict[str, float]:
        numeric_fields = {
            "total_time": self.total_time,
            "best_score": self.best_score,
            "n_trials": float(self.n_trials),
            "n_successful_trials": float(self.n_successful_trials),
            "peak_memory_gb": self.peak_memory_gb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "avg_trial_time": self.avg_trial_time,
            "max_trial_time": self.max_trial_time,
            "latest_memory_gb": self.latest_memory_gb,
        }
        return {key: float(value) for key, value in numeric_fields.items()}


# =====================================================================================
# 🔧 CAMADA DE APLICAÇÃO - CASOS DE USO, PROTOCOLS E OBSERVERS
# =====================================================================================


class OptimizationEvent(Enum):
    (
        OPTIMIZATION_START,
        DATA_VALIDATION_START,
        DATA_VALIDATION_COMPLETE,
        OPTIMIZATION_METHOD_START,
        OPTIMIZATION_METHOD_COMPLETE,
        BEST_MODEL_SAVED,
        SENSITIVITY_ANALYSIS_COMPLETE,
        OPTIMIZATION_COMPLETE,
        OPTIMIZATION_FAILED,
    ) = range(9)


@runtime_checkable
class OptimizationObserver(Protocol):
    def update(self, event: OptimizationEvent, data: Dict[str, Any]): ...


class Subject:
    def __init__(self):
        self._observers: List[OptimizationObserver] = []

    def attach(self, observer: OptimizationObserver):
        self._observers.append(observer)

    def notify(self, event: OptimizationEvent, data: Dict[str, Any]):
        for observer in self._observers:
            observer.update(event, data)


@runtime_checkable
class OptimizationStrategy(Protocol):
    def optimize(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, config: Dict[str, Any]
    ) -> Dict[str, Any]: ...
    def save_best_model(
        self, model: Any, params: Dict[str, Any], score: float, path: str
    ) -> Path: ...
    def _generate_sensitivity_analysis(self, output_dir: Path) -> Path: ...
    @property
    def best_model(self) -> Any: ...


# =====================================================================================
# 🏭 CAMADA DE INFRAESTRUTURA - OBSERVERS, ESTRATÉGIAS E FACTORY
# =====================================================================================


class ConsoleLogObserver(OptimizationObserver):
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger

    def update(self, event: OptimizationEvent, data: Dict[str, Any]):
        method = data.get("method")
        method_val = method.value if isinstance(method, Enum) else "N/A"
        resource_usage = data.get("resource_usage", {})
        messages = {
            OptimizationEvent.OPTIMIZATION_START: "🚀 === INICIANDO OTIMIZAÇÃO DE HIPERPARÂMETROS ===",
            OptimizationEvent.OPTIMIZATION_METHOD_START: f"🎯 EXECUTANDO MÉTODO: {method_val.upper()}",
            OptimizationEvent.BEST_MODEL_SAVED: f"💾 Melhor modelo salvo: {data.get('model_path', 'N/A')}",
            OptimizationEvent.OPTIMIZATION_COMPLETE: (
                "🎉 OTIMIZAÇÃO CONCLUÍDA em "
                f"{data.get('total_time', 0):.2f}s | Pico RAM: "
                f"{resource_usage.get('peak_memory_gb', 0):.2f}GB | Média trial: "
                f"{resource_usage.get('avg_trial_time', 0):.2f}s"
            ),
            OptimizationEvent.OPTIMIZATION_FAILED: f"❌ ERRO CRÍTICO NA OTIMIZAÇÃO: {data.get('error', 'Desconhecido')}",
        }
        if message := messages.get(event):
            self.logger.log(logging.INFO, message)


class MLflowObserver(OptimizationObserver):
    def __init__(self, experiment_name: str, project_root: Path):
        self.experiment_name, self.project_root = experiment_name, project_root
        self.enabled = mlflow is not None
        if not self.enabled:
            logging.getLogger("TrustShield-Optimizer").warning(
                "MLflow não disponível - Observer desativado."
            )

    def update(self, event: OptimizationEvent, data: Dict[str, Any]):
        if not self.enabled or mlflow is None:
            return

        if event == OptimizationEvent.OPTIMIZATION_START:
            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run(
                run_name=f"optimization_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            mlflow.log_params(data.get("config", {}).get("hyper_optimization", {}))
        elif event == OptimizationEvent.OPTIMIZATION_METHOD_COMPLETE:
            metrics: OptimizationMetrics = data["metrics"]
            mlflow.log_metrics(metrics.to_numeric_metrics())
            mlflow.set_tag("optimization_method", metrics.method.value)
            mlflow.set_tag("metrics_timestamp", metrics.timestamp.isoformat())
        elif event == OptimizationEvent.BEST_MODEL_SAVED:
            if model_path := data.get("model_path"):
                model_artifact = Path(model_path)
                if model_artifact.is_file():
                    mlflow.log_artifact(str(model_artifact))
        elif event == OptimizationEvent.SENSITIVITY_ANALYSIS_COMPLETE:
            if sensitivity_path := data.get("sensitivity_path"):
                if Path(sensitivity_path).exists():
                    mlflow.log_artifact(str(sensitivity_path))
        elif event == OptimizationEvent.OPTIMIZATION_COMPLETE:
            mlflow.set_tag("status", "success")
            mlflow.end_run()
        elif event == OptimizationEvent.OPTIMIZATION_FAILED:
            if mlflow.active_run():
                mlflow.set_tag("status", "failed")
                mlflow.end_run(status="FAILED")


class BaseOptimizationStrategy:
    def __init__(
        self, config: Dict[str, Any], logger: AdvancedLogger, monitor: ResourceMonitor
    ):
        self.config = config
        self.logger = logger
        self.monitor = monitor
        self._best_score: float = -np.inf
        self._best_params: Dict[str, Any] = {}
        self._best_model: Any = None

    @property
    def best_model(self) -> Any:
        return self._best_model

    def _objective_function(
        self, params: Dict[str, Any], X_train: pd.DataFrame, X_val: pd.DataFrame
    ) -> float:
        trial_start = time.time()
        self.monitor.update_peak_memory()
        try:
            model_params = dict(params)
            model_params.setdefault("random_state", 42)
            model = IsolationForest(**model_params)
            model.fit(X_train)
            self.monitor.update_peak_memory()
            scores = -model.decision_function(X_val)
            self.monitor.update_peak_memory()
            objective_value = float(np.var(scores))
            if objective_value > self._best_score:
                self._best_score = objective_value
                self._best_params = params.copy()
                self._best_model = model
            return objective_value
        except Exception as e:
            self.logger.log(logging.WARNING, f"Erro na avaliação de parâmetros: {e}")
            return -np.inf
        finally:
            self.monitor.record_trial_time(time.time() - trial_start)
            self.monitor.update_peak_memory()

    def save_best_model(
        self, model: Any, params: Dict[str, Any], score: float, path: str
    ) -> Path:
        self._best_model, self._best_params, self._best_score = model, params, score
        return self._save_best_model_artifact(Path(path))

    def _save_best_model_artifact(self, output_dir: Path) -> Path:
        if self._best_model is None:
            self.logger.log(
                logging.WARNING, "Nenhum modelo válido encontrado para salvar."
            )
            return Path()
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = output_dir / f"best_model_{timestamp}.joblib"
            artifact = {
                "model": self._best_model,
                "params": self._best_params,
                "score": self._best_score,
                "timestamp": timestamp,
            }
            joblib.dump(artifact, model_path)
            self.logger.log(logging.INFO, f"Melhor modelo salvo: {model_path}")
            return model_path
        except Exception as e:
            self.logger.log(logging.ERROR, f"Falha ao salvar melhor modelo: {e}")
            raise

    def _generate_sensitivity_analysis(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = (
            output_dir
            / f"sensitivity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        html_content = (
            f"<h1>Sensitivity Analysis for best score: {self._best_score:.4f}</h1>"
        )
        with open(report_path, "w") as f:
            f.write(html_content)
        return report_path


class OptunaOptimizer(BaseOptimizationStrategy, OptimizationStrategy):
    def optimize(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        start_time = time.time()
        self.logger.log(logging.INFO, "Iniciando otimização com Optuna...")
        if not OPTUNA_AVAILABLE or optuna is None:
            raise RuntimeError(
                "Optuna não está disponível neste ambiente. Instale 'optuna' para executar a otimização."
            )

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        search_space = config["hyper_optimization"]["space"]

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", *search_space["n_estimators"]
                ),
                "max_samples": trial.suggest_float(
                    "max_samples", *search_space["max_samples"]
                ),
                "max_features": trial.suggest_float(
                    "max_features", *search_space["max_features"]
                ),
                "contamination": trial.suggest_float(
                    "contamination", *search_space.get("contamination", [0.01, 0.5])
                ),
                "n_jobs": -1,
            }
            return self._objective_function(params, X_train, X_val)

        n_trials = config["hyper_optimization"].get("n_trials", 50)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        successful_trials = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        )
        metrics = OptimizationMetrics(
            method=OptimizationMethod.OPTUNA,
            total_time=time.time() - start_time,
            best_score=study.best_value,
            n_trials=n_trials,
            n_successful_trials=successful_trials,
            peak_memory_gb=self.monitor.peak_memory,
            cpu_usage_percent=psutil.cpu_percent(interval=0.1),
            avg_trial_time=self.monitor.average_trial_time,
            max_trial_time=self.monitor.max_trial_time,
            latest_memory_gb=self.monitor.latest_memory_gb,
        )
        self.logger.log(
            logging.INFO,
            f"Optuna concluído: {n_trials} trials, melhor score: {study.best_value:.4f}",
        )
        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "metrics": metrics,
        }


class RayTuneOptimizer(BaseOptimizationStrategy, OptimizationStrategy):
    def optimize(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not (RAY_AVAILABLE and ray and AsyncHyperBandScheduler and OptunaSearch):
            raise ImportError(
                "Ray Tune ou suas dependências não estão instalados. Instale com: pip install ray[tune]"
            )
        # ... (implementação completa da otimização com Ray Tune) ...
        return {}  # Placeholder


class OptimizationStrategyFactory:
    @staticmethod
    def create_strategy(
        method: OptimizationMethod,
        config: Dict[str, Any],
        logger: AdvancedLogger,
        monitor: ResourceMonitor,
    ) -> OptimizationStrategy:
        strategies = {
            OptimizationMethod.OPTUNA: OptunaOptimizer,
            OptimizationMethod.RAY_TUNE: RayTuneOptimizer,
        }
        strategy_class = strategies.get(method)
        if not strategy_class:
            raise ValueError(f"Estratégia de otimização não encontrada: {method.value}")
        return strategy_class(config, logger, monitor)


# =====================================================================================
# 🎼 ORQUESTRADOR - O SERVIÇO PRINCIPAL DA APLICAÇÃO
# =====================================================================================


class ResilientHyperparameterOptimizer(Subject):
    def __init__(self, data_path: str, config_path: str = "config/config.yaml"):
        super().__init__()
        self.project_root = Path(__file__).resolve().parents[2]
        self.data_path = Path(data_path)
        self.config_path = config_path
        self.logger = AdvancedLogger("TrustShield-Optimizer")
        self.monitor = ResourceMonitor(self.logger)
        self.config_manager = ConfigManager(self.project_root, self.logger)
        self.config = self.config_manager.get_config()
        self.data_validator = DataValidator(self.config, self.logger)
        self.attach(ConsoleLogObserver(self.logger))
        self.attach(
            MLflowObserver(
                self.config.get("mlflow", {}).get("experiment_name", "TrustShield"),
                self.project_root,
            )
        )

    @circuit(failure_threshold=3, recovery_timeout=30)
    def run_optimization(self, methods: List[str] = None):
        self.monitor.reset()
        start_time = time.time()
        try:
            self.notify(
                OptimizationEvent.OPTIMIZATION_START,
                {"data_path": str(self.data_path), "config": self.config},
            )
            self.notify(OptimizationEvent.DATA_VALIDATION_START, {})
            X_train, X_val = self._load_and_validate_data()
            self.notify(
                OptimizationEvent.DATA_VALIDATION_COMPLETE,
                {
                    "validation_status": "success",
                    "train_samples": len(X_train),
                    "val_samples": len(X_val),
                },
            )
            if methods is None:
                methods = self.config.get("hyper_optimization", {}).get(
                    "methods", ["optuna"]
                )
            optimization_methods = [
                OptimizationMethod(m)
                for m in methods
                if m in OptimizationMethod._value2member_map_
            ]
            results = {}
            for method in optimization_methods:
                self.notify(
                    OptimizationEvent.OPTIMIZATION_METHOD_START, {"method": method}
                )
                strategy = OptimizationStrategyFactory.create_strategy(
                    method, self.config, self.logger, self.monitor
                )
                optimization_results = strategy.optimize(X_train, X_val, self.config)
                output_dir = (
                    self.project_root / "outputs" / "optimization" / method.value
                )
                model_path = strategy.save_best_model(
                    optimization_results.get("best_model", strategy.best_model),
                    optimization_results["best_params"],
                    optimization_results["best_score"],
                    str(output_dir),
                )
                sensitivity_path = strategy._generate_sensitivity_analysis(output_dir)
                results[method.value] = optimization_results
                self.notify(
                    OptimizationEvent.OPTIMIZATION_METHOD_COMPLETE,
                    {"method": method, "metrics": optimization_results["metrics"]},
                )
                if model_path.is_file():
                    self.notify(
                        OptimizationEvent.BEST_MODEL_SAVED, {"model_path": model_path}
                    )
                if sensitivity_path.exists():
                    self.notify(
                        OptimizationEvent.SENSITIVITY_ANALYSIS_COMPLETE,
                        {"sensitivity_path": sensitivity_path},
                    )
            self.notify(
                OptimizationEvent.OPTIMIZATION_COMPLETE,
                {
                    "total_time": time.time() - start_time,
                    "methods_used": [m.value for m in optimization_methods],
                    "best_overall_score": max(
                        r["best_score"] for r in results.values()
                    ),
                    "resource_usage": {
                        "peak_memory_gb": self.monitor.peak_memory,
                        "avg_trial_time": self.monitor.average_trial_time,
                        "max_trial_time": self.monitor.max_trial_time,
                        "latest_memory_gb": self.monitor.latest_memory_gb,
                    },
                },
            )
        except Exception as e:
            self.notify(OptimizationEvent.OPTIMIZATION_FAILED, {"error": str(e)})
            self.logger.log(logging.ERROR, f"Erro na otimização: {e}", exc_info=True)
            raise
        finally:
            self.monitor.update_peak_memory()

    def _load_and_validate_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        full_data_path = (
            self.project_root / self.data_path
            if not self.data_path.is_absolute()
            else self.data_path
        )
        self.logger.log(logging.INFO, f"Carregando dados de: {full_data_path}")
        if full_data_path.suffix == ".parquet":
            data = pd.read_parquet(full_data_path)
        elif full_data_path.suffix == ".csv":
            data = pd.read_csv(full_data_path)
        else:
            raise ValueError(
                f"Formato de arquivo não suportado: {full_data_path.suffix}"
            )
        validation_success, _ = self.data_validator.validate(data)
        if not validation_success:
            raise ValueError("Validação de dados falhou.")
        features = data.copy()
        target_column = (
            self.config.get("training", {}).get("target_column") or "is_anomaly"
        )
        target = None
        if target_column in features.columns:
            target = features[target_column]
            features = features.drop(columns=[target_column])

        datetime_cols = [
            col
            for col in features.columns
            if pd.api.types.is_datetime64_any_dtype(features[col])
        ]
        for col in datetime_cols:
            converted = pd.to_datetime(features[col], utc=True, errors="coerce")
            features[col] = converted.map(
                lambda x: x.timestamp() if not pd.isna(x) else np.nan
            )

        non_numeric_cols = features.select_dtypes(exclude=[np.number]).columns
        if non_numeric_cols.any():
            removed_cols = list(non_numeric_cols)
            self.logger.log(
                logging.WARNING,
                f"Removendo colunas não numéricas antes da otimização: {removed_cols}",
            )
            features = features.drop(columns=removed_cols)

        if features.empty:
            raise ValueError(
                "Nenhuma coluna numérica disponível para otimização após pré-processamento."
            )

        features = features.fillna(features.mean(numeric_only=True))

        stratify_target = None
        if target is not None:
            if target.nunique(dropna=True) > 1:
                non_null_target = target.dropna()
                replacement = non_null_target.mode().iloc[0] if not non_null_target.empty else 0
                stratify_target = target.fillna(replacement)

        X_train, X_val = train_test_split(
            features, test_size=0.2, random_state=42, stratify=stratify_target
        )
        numeric_cols = X_train.select_dtypes(include=np.number).columns
        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
        self.logger.log(
            logging.INFO,
            f"Dados preparados: {len(X_train)} treino, {len(X_val)} validação",
        )
        return X_train, X_val


# =====================================================================================
# 🚀 PONTO DE ENTRADA DA APLICAÇÃO
# =====================================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Sistema de Otimização de Hiperparâmetros TrustShield"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Caminho para os dados de otimização"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["optuna"],
        help="Métodos de otimização (optuna, ray_tune)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Arquivo de configuração",
    )
    args = parser.parse_args()
    try:
        optimizer = ResilientHyperparameterOptimizer(
            data_path=args.data, config_path=args.config
        )
        optimizer.run_optimization(methods=args.methods)
        sys.exit(0)
    except Exception as e:
        print(f"❌ ERRO CRÍTICO NA EXECUÇÃO: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
