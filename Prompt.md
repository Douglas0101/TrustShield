# Auditoria Técnica Profunda — TrustShield (branch `commit-2`)

> Escopo: análise técnica do repositório, arquitetura, MLOps, segurança, observabilidade e prontidão para produção; identificação de gaps; recomendações priorizadas com exemplos de configuração/código.

---

## 1) Visão Geral

**Resumo do projeto**: Plataforma de detecção e prevenção de fraudes com foco em **aprendizado não supervisionado** (modelo campeão: *Isolation Forest*), governança via **MLflow**, armazenamento de artefatos em **MinIO** com backend **PostgreSQL**, conteinerização Docker e orquestração via *Makefile* e *docker-compose*. Arquitetura declarada como **Hexagonal (Ports & Adapters)** com comunicação **orientada a eventos**.

**Principais pastas e artefatos (observados na raiz):** `config/`, `docker/`, `mlruns/`, `notebooks/`, `outputs/`, `secrets/`, `src/`, `tests/`, `Makefile`, `requirements.txt`, documentos PDF (arquitetura, engenharia, avaliação de modelos, roadmap), além de arquivos de metodologia e comandos Docker.

---

## 2) Pontos Fortes

* **Intenção de Arquitetura Hexagonal (DDD)**: separação de domínio, aplicação e infraestrutura, favorecendo testabilidade e desacoplamento.
* **Pilares de MLOps bem escolhidos**: MLflow + MinIO + Postgres cobrem rastreabilidade, versionamento e storage agnóstico (S3-compatível).
* **Automação com Makefile** para ciclo *build → run → logs → stop*, reduzindo atrito do setup.
* **Métricas de SLO/SLA definidas** (p.ex. p95 < 200 ms, disponibilidade 99.9%) que orientam decisões de engenharia.
* **Testes previstos** (unitários, integração, BDD, propriedades) — boa direção para robustez.

---

## 3) Riscos e Gaps Técnicos

> Itens priorizados por severidade (Alta/Moderada/Baixa) e esforço (★=baixo … ★★★=alto)

1. **Artefatos versionados no Git (Alta, ★)**: presença de `mlruns/`, `outputs/`, possivelmente `cache/` e `Saida_Algoritmo.txt`. Isso incha o repo, quebra reprodutibilidade (artefatos locais) e pode vazar dados sensíveis.
2. **Diretório `secrets/` no repositório (Alta, ★)**: risco de exposição de credenciais. Mesmo vazio, induz mau padrão; se populado, é incidente esperando acontecer.
3. **Stack “orientada a eventos” sem broker explícito (Moderada, ★★)**: README fala em eventos, mas o stack não lista **Kafka/NATS/RabbitMQ**. Sem um broker, sua arquitetura “event-driven” fica só conceitual.
4. **`requirements.txt` possivelmente não *pinned* (Moderada, ★)**: sem *pinning* (ou lockfile), reprodutibilidade e segurança sofrem; builds flutuam com versões do PyPI.
5. **Ausência de CI/CD visível (Moderada, ★★)**: sem pipelines públicos (GH Actions) para lint, testes, *image build* e *security scan*.
6. **Falta de orquestrador de jobs** (Moderada, ★★): *Makefile* + `docker-compose` atendem dev, mas produção pede **Prefect/Airflow/Argo** para DAGs versionadas, *retries*, *backfills* e SLAs.
7. **Observabilidade incompleta (Moderada, ★★)**: sem evidência de *metrics* e *tracing* (Prometheus/OpenTelemetry) em API e jobs.
8. **Monitoramento de dados/modelo (Moderada, ★★)**: faltam *data checks*, *drift*, *performance decay*. PDFs de análise ajudam, porém precisam virar rotinas automatizadas.
9. **Segurança de container (Moderada, ★)**: sem Dockerfile visível aqui, mas checklist indica pontos comuns: execução como `root`, imagens não *slim*, sem *SBOM*, sem *scan* (Trivy/Grype).
10. **Qualidade de código/estilo (Baixa, ★)**: consolidar *tooling* (ruff/black/isort/mypy/pre-commit) e *conventional commits*.

---

## 4) Arquitetura Recomendada (alvo)

### 4.1 Macro (serviços)

* **Ingestion** (stream/batch): conecta gateway de pagamentos, fila/broker (Kafka) e *feature store*.
* **Feature Service**: computa/serve *features* (online/offline) — considerar **Feast**.
* **Training Orchestrator** (Prefect/Airflow): DAGs para EDA → limpeza → *feature build* → treino → avaliação → registro em MLflow → *promotion*.
* **Model Registry** (MLflow) + **Artifact Store** (MinIO).
* **Inference API (FastAPI/Uvicorn)** stateless, *autoscaling*; *batch scorer* assíncrono para reprocessos.
* **Monitoring**: Prometheus/Grafana + OpenTelemetry + *model monitoring* (Evidently/NannyML) + *data quality* (Great Expectations).

### 4.2 Ports & Adapters (exemplo de pastas)

```
src/
  domain/
    entities.py           # Transaction, Card, Account, FraudScore
    value_objects.py      # Money, MerchantCategory, Geo
    services.py           # FraudPolicy (score thresholds, hold/release)
  application/
    use_cases/
      score_transaction.py
      retrain_model.py
    dto.py                # Request/Response DTOs (pydantic)
  infrastructure/
    brokers/kafka.py
    storage/minio_store.py
    storage/pg_repository.py
    model_registry/mlflow.py
    api/http_fastapi.py
    features/feast_provider.py
```

---

## 5) MLOps — Fluxos e Boas Práticas

### 5.1 Treino & Registro

* **Isolation Forest**: parametrizar e registrar **`n_estimators`**, **`max_samples`**, **`contamination`**, **`max_features`**, **`bootstrap`**, **`random_state`** e **`n_jobs=-1`** para paralelismo CPU.
* **Reprodutibilidade**: *seed* global; *pip-compile* (hashes) ou Poetry para *locking*; `MLFLOW_TRACKING_URI`, `MLFLOW_S3_ENDPOINT_URL` e credenciais via *env vars*.
* **Artefatos**: apenas no MinIO. **Não** versionar `mlruns/`/`models/` no Git.

### 5.2 Feature Engineering

* *Pipelines* idempotentes. **Skew** offline/online mitigado com biblioteca única para *feature transforms* (ex.: `src/features/transformers.py`).
* **Feature Store** (Feast) para servir *features* consistentes (OLTP/OLAP), com *materialization* para baixa latência.

### 5.3 Avaliação

* Como é *unsupervised*, use *proxy labels* e *scenario-based tests*: injete anomalias sintéticas; avalie **ROC/PR** sobre *silver labels*, *Precision\@K* e *Alert FP Rate*.
* **Backtesting** por janela temporal (rolling); *model selection* com *tolerance bands* sobre métricas, não só *point estimates*.

### 5.4 Deploy & Rollback

* *Blue/Green* ou *Canary* com dois modelos no registro ("Champion/Challenger"); *feature flags* para *routing*.
* Expor *model version*, *git sha*, *build date* no `/health` e `/metrics`.

---

## 6) Segurança (App, Dados, Supply Chain)

**Credenciais & Segredos**

* Remover `secrets/` do repo; usar **Vault** (HashiCorp), **AWS Secrets Manager** ou **Doppler**. Injetar via `env`/K8s **Secret**.

**Dependências**

* Fixar versões (`pip-compile`), gerar **SBOM** (Syft) e rodar *scans* (Grype/Trivy). Habilitar **Dependabot**.

**Containers**

* *Multi-stage builds*, usuário não-root, `distroless`/`slim`, *`--cap-drop ALL`*, `read-only` FS, *healthchecks*.

**Dados sensíveis (LGPD)**

* Classificar PII (nome, CPF, cartão, geoloc). *Tokenização* de PAN, *KMS* para chaves, *data retention* e *purpose limitation*.

**Model Security**

* Defesa contra *evasion* e *data poisoning*: *adversarial validation*, *noise stress*, *rate limiting* e *shadow testing*.

---

## 7) Observabilidade & Confiabilidade

* **Métricas técnicas**: latência p50/p95/p99, RPS, erro 4xx/5xx, fila (lag do Kafka), uso CPU/RAM.
* **Métricas de negócio**: TP/FP/FN por segmento, perda evitada, *alert volume* por hora.
* **Tracing distribuído**: OpenTelemetry → OTLP → Tempo/Grafana. Propagar *trace-id* do gateway até scoring.
* **Data/Model Monitoring**: Evidently (drift covariate/target), NannyML (CBPE), Great Expectations (contratos de schema) no ingestion.

---

## 8) Testes (pirâmide e exemplos)

* **Unitários**: *domain services* (puro Python), *feature transformers* (determinismo), utilitários (hashing de *features*).
* **Contratos**: Pact (Producer/Consumer) entre API e clientes.
* **Integração**: subir *stack* mínima via `docker-compose -f docker/docker-compose.yml` e executar *e2e* de um lote sintético → `MLflow run` → *registry*.
* **Propriedades**: Hypothesis para invariantes (ex.: `score ∈ [0,1]`, monotonicidade de thresholds).
* **BDD**: cenários de fraude legítima vs. *chargeback*, *card-not-present*, *device spoofing*.

---

## 9) DevEx & Governança

* **Pre-commit** com *ruff*, *black*, *isort*, *mypy*, *bandit*, *nbstripout* (limpar *outputs* de notebooks).
* **Conventional Commits** + *semantic-release* → *CHANGELOG* e *tags*.
* **CODEOWNERS**, **CONTRIBUTING.md**, *PR template*, *issue templates*.
* **Branching**: *trunk-based* com *short-lived branches* e *feature flags*.

---

## 10) Exemplos (trechos prontos)

### 10.1 `pyproject.toml` (tooling básico)

```toml
[tool.black]
line-length = 100

[tool.isort]
profile = "black"

[tool.ruff]
line-length = 100
select = ["E","F","I","B","UP","S","PL"]

[tool.mypy]
python_version = "3.10"
strict = true
warn_unused_ignores = true

[tool.bandit]
targets = ["src"]
```

### 10.2 Workflow CI (GitHub Actions)

```yaml
name: ci
on: [push, pull_request]
jobs:
  build-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.10' }
      - name: Cache pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
      - run: pip install pip-tools
      - run: pip-compile -q requirements.in -o requirements.txt || true
      - run: pip install -r requirements.txt
      - run: pip install ruff black mypy bandit pytest pytest-cov
      - run: ruff check src
      - run: black --check src tests
      - run: mypy src
      - run: bandit -r src -x tests
      - run: pytest -q --maxfail=1 --disable-warnings --cov=src
```

### 10.3 `docker/docker-compose.yml` (mínimo sugerido)

```yaml
version: '3.9'
services:
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
    ports: ["9000:9000", "9001:9001"]
    volumes: ["minio_data:/data"]

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: mlflow
    ports: ["5432:5432"]
    volumes: ["pg_data:/var/lib/postgresql/data"]

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.14.1
    command: >
      mlflow server --host 0.0.0.0 --port 5000 \
      --backend-store-uri postgresql+psycopg2://postgres:${POSTGRES_PASSWORD}@postgres:5432/mlflow \
      --artifacts-destination s3://mlflow/
    environment:
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
      AWS_ACCESS_KEY_ID: ${MINIO_ROOT_USER}
      AWS_SECRET_ACCESS_KEY: ${MINIO_ROOT_PASSWORD}
    ports: ["5000:5000"]
    depends_on: [minio, postgres]

  trainer:
    build: ../
    command: python -m src.models.train_fraud_model
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
      AWS_ACCESS_KEY_ID: ${MINIO_ROOT_USER}
      AWS_SECRET_ACCESS_KEY: ${MINIO_ROOT_PASSWORD}
    depends_on: [mlflow]

volumes: { minio_data: {}, pg_data: {} }
```

### 10.4 `src/api/http_fastapi.py` (esqueleto)

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

class Txn(BaseModel):
    amount: float
    merchant_cat: int
    country: str
    device_score: float

model = None

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("/models/isoforest.joblib")

@app.post("/score")
def score(txn: Txn):
    # TODO: mesma lógica de featurização usada no treino
    features = [[txn.amount, txn.merchant_cat, txn.device_score]]
    anomaly_score = -float(model.decision_function(features)[0])
    return {"anomaly_score": anomaly_score}
```

---

## 11) Checklist de Hardening

* [ ] Remover `mlruns/`, `outputs/`, `cache/` e arquivos de saída do Git; adicionar ao `.gitignore`.
* [ ] Eliminar `secrets/` do repo; mover segredos para gerenciador apropriado.
* [ ] Fixar dependências e ativar *pip-audit* e *bandit* em CI.
* [ ] Criar *pre-commit* com ruff/black/isort/mypy/nbstripout.
* [ ] Provisionar Kafka (ou NATS) para cumprir *event-driven*.
* [ ] Implementar Observabilidade (Prometheus + OTel) e dashboards de fraude/negócio.
* [ ] Orquestrador (Prefect/Airflow) e DAGs declarativas do pipeline.
* [ ] Testes de contrato (Pact), *e2e* com `docker-compose`, hipóteses e BDD.
* [ ] Políticas LGPD: classificação PII, tokenização, *retention* e trilhas de auditoria.

---

## 12) Roadmap 0–90 dias (priorizado)

**0–14 dias**

* Higiene do repositório (itens de hardening acima).
* CI simples (lint, testes, build Docker, scans). *Owners* e *templates*.
* Unificar pipelines de *features* para inferência e treino.

**15–45 dias**

* Kafka/NATS e *ingestion services*.
* Observabilidade ponta-a-ponta; métricas de negócio.
* Orquestrador (Prefect) com *schedules* e *retries*.

**46–90 dias**

* Feature Store (Feast) com materialização.
* Canary deploy de modelos + *champion/challenger*.
* Monitoramento de drift e *alerting*; rotinas de *auto-retrain* controladas por *guardrails*.

---

## 13) Conclusão

O TrustShield tem **fundação sólida** (MLflow, MinIO, Postgres, Docker) e **boas intenções arquiteturais** (Hexagonal/DDD, SLOs definidos). Para chegar a produção *enterprise*, foque nos **gaps de segurança**, **observabilidade real**, **event broker concreto**, **orquestração de jobs**, **pinning de dependências** e **higiene do repositório**. As seções de exemplos e o checklist fornecem um caminho objetivo para reduzir risco e acelerar *time-to-prod*.
