from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from types import ModuleType, SimpleNamespace

import yaml

try:  # pragma: no cover - dependência opcional
    import boto3  # type: ignore
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore

try:  # pragma: no cover - dependência opcional
    import mlflow  # type: ignore
    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore
    MLFLOW_AVAILABLE = False


class _NoOpMlflow:
    """Implementa uma API mínima do MLflow para ambientes sem a dependência."""

    def __init__(self):
        self._tracking_uri = "noop://mlflow"
        self.tracking = SimpleNamespace(MlflowClient=lambda *args, **kwargs: None)
        self.sklearn = SimpleNamespace(log_model=self._noop, autolog=self._noop)

    def _noop(self, *args, **kwargs):  # pragma: no cover - sem efeitos colaterais
        return None

    def set_tracking_uri(self, uri: str) -> None:
        self._tracking_uri = uri

    def get_tracking_uri(self) -> str:
        return self._tracking_uri

    def set_experiment(self, *args, **kwargs) -> None:
        pass

    def start_run(self, *args, **kwargs):
        return nullcontext()

    def log_params(self, *args, **kwargs) -> None:
        pass

    def log_metrics(self, *args, **kwargs) -> None:
        pass

    def set_tag(self, *args, **kwargs) -> None:
        pass

    def set_tags(self, *args, **kwargs) -> None:
        pass

    def end_run(self, *args, **kwargs) -> None:
        pass

    def log_artifact(self, *args, **kwargs) -> None:
        pass

    def active_run(self):
        return None


if not MLFLOW_AVAILABLE:
    _mlflow_impl = _NoOpMlflow()

    def _wrap(name):
        def _inner(*args, **kwargs):
            return getattr(_mlflow_impl, name)(*args, **kwargs)

        return _inner

    _mlflow_module = ModuleType("mlflow")
    for attr_name in [
        "set_tracking_uri",
        "get_tracking_uri",
        "set_experiment",
        "start_run",
        "log_params",
        "log_metrics",
        "set_tag",
        "set_tags",
        "end_run",
        "log_artifact",
        "active_run",
    ]:
        setattr(_mlflow_module, attr_name, _wrap(attr_name))

    _mlflow_module.tracking = _mlflow_impl.tracking
    _mlflow_module.sklearn = _mlflow_impl.sklearn
    sys.modules.setdefault("mlflow", _mlflow_module)
    mlflow = _mlflow_module  # type: ignore

_DEF_S3_DOCKER = "http://trustshield_minio:9000"
_DEF_S3_HOST = "http://localhost:9000"
_DEF_BUCKET = "mlflow"


def _coalesce(*vals: Optional[str]) -> Optional[str]:
    for v in vals:
        if v:
            return v
    return None


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_config() -> Dict[str, Any]:
    """Carrega a configuração principal do projeto."""
    root = _get_project_root()
    config_path = root / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado em: {config_path}"
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_mlflow(
    experiment: str = "TrustShield",
    tracking_uri: Optional[str] = None,
    s3_endpoint: Optional[str] = None,
    ensure_bucket: bool = True,
    bucket_name: str = _DEF_BUCKET,
) -> mlflow:
    """
    Configura o MLflow usando o `config.yaml` como fonte de verdade.
    A ordem de precedência para a URI de tracking é:
    1. Argumento `tracking_uri` passado para a função.
    2. Variável de ambiente `MLFLOW_TRACKING_URI`.
    3. Valor de `mlflow.tracking_uri` no `config.yaml`.
    4. Fallback para localhost se nada for encontrado.
    """
    if not MLFLOW_AVAILABLE:
        print(
            "⚠️  MLflow não está instalado. Será utilizado um stub sem efeitos colaterais."
        )

    config = _load_config()
    mlflow_config = config.get("mlflow", {})

    # Determina a URI de Tracking
    config_uri = mlflow_config.get("tracking_uri")
    final_tracking_uri = _coalesce(
        tracking_uri,
        os.getenv("MLFLOW_TRACKING_URI"),
        config_uri,
        "http://localhost:5000",
    )

    # Determina o endpoint S3
    in_c = (
        "docker" in open("/proc/1/cgroup").read()
        or "containerd" in open("/proc/1/cgroup").read()
    )
    final_s3_endpoint = _coalesce(
        s3_endpoint,
        os.getenv("MLFLOW_S3_ENDPOINT_URL"),
        _DEF_S3_DOCKER if in_c else _DEF_S3_HOST,
    )

    os.environ["MLFLOW_TRACKING_URI"] = final_tracking_uri
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = final_s3_endpoint

    mlflow.set_tracking_uri(final_tracking_uri)

    # Usa o nome do experimento do config.yaml se não for passado um específico
    final_experiment_name = (
        experiment
        if experiment != "TrustShield"
        else mlflow_config.get("experiment_name", "TrustShield")
    )
    mlflow.set_experiment(final_experiment_name)

    if ensure_bucket and boto3 is not None:
        _ensure_minio_bucket(bucket_name, final_s3_endpoint)

    print(
        f"MLflow setup complete. Tracking URI: {final_tracking_uri}, Experiment: {final_experiment_name}"
    )

    return mlflow


def _ensure_minio_bucket(bucket_name: str, endpoint_url: str) -> None:
    try:
        import boto3
        from botocore.exceptions import ClientError

        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )
        try:
            s3.head_bucket(Bucket=bucket_name)
            return
        except ClientError:
            pass
        s3.create_bucket(Bucket=bucket_name)
    except Exception:
        pass


def enable_sklearn_autolog(log_models: bool = True) -> None:
    try:
        from mlflow import sklearn as mlflow_sklearn  # type: ignore

        mlflow_sklearn.autolog(log_models=log_models)
    except Exception:
        if hasattr(mlflow, "sklearn"):
            try:
                mlflow.sklearn.autolog(log_models=log_models)  # type: ignore[attr-defined]
            except Exception:
                pass
        else:
            pass


def log_params_flat(
    d: Dict[str, Any], prefix: str = "", allow: Iterable[type] = (str, int, float, bool)
) -> None:
    if not d:
        return
    mlflow.log_params({f"{prefix}{k}": v for k, v in d.items() if isinstance(v, allow)})


def log_json(
    obj: Any, artifact_path: str = "artifacts", filename: str = "data.json"
) -> Path:
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        tmp = Path(f.name)
    mlflow.log_artifact(str(tmp), artifact_path=artifact_path)
    return tmp


def log_figure(
    fig, artifact_path: str = "figures", filename: Optional[str] = None
) -> Path:
    if filename is None:
        filename = f"figure_{int(time.time())}.png"
    out = Path.cwd() / filename
    fig.savefig(out, bbox_inches="tight")
    mlflow.log_artifact(str(out), artifact_path=artifact_path)
    return out


@contextmanager
def mlflow_run(run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    if run_name is None:
        run_name = f"run_{int(time.time())}"
    mlflow.start_run(run_name=run_name)
    if tags:
        mlflow.set_tags(tags)
    try:
        yield
        mlflow.set_tag("status", "success")
        mlflow.end_run()
    except Exception:
        try:
            mlflow.set_tag("status", "failed")
        finally:
            mlflow.end_run(status="FAILED")
        raise
