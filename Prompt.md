# Descrição do problema (comportamento observado)

* O comando **`make up`** constrói com êxito as imagens `trustshield-api` e `trustshield-dashboard` (builds finalizam sem erros).
* Na fase **`docker compose up -d`**, os serviços são criados e iniciados; **MinIO** reporta **Healthy** e o `minio-init` conclui (**Exited**).
* O contêiner **`trustshield-mlflow-1`** entra no estado **Error/Unhealthy** após \~134 s. Em consequência, o Compose interrompe o bootstrap e retorna:

  > dependency failed to start: container trustshield-mlflow-1 is unhealthy
  > O alvo `up` do Makefile falha com **`Erro 1`** (linha 45), e a pilha não fica de pé.

# Precisão do escopo (o que está e o que não está falhando)

* **Funciona:** build das imagens; subida do **MinIO** (saúde OK) e finalização do **minio-init**.
* **Falha:** **saúde do serviço MLflow** em runtime (não é erro de build). O Compose **bloqueia** por política de dependência/healthcheck (provavelmente `depends_on: condition: service_healthy`), portanto outros serviços ficam **Created/Recreated** mas a orquestração é abortada.

# Sintoma raiz imediato

* **Causa imediata (não a causa técnica profunda):** o **healthcheck do `trustshield-mlflow-1`** não atinge estado **healthy** no prazo; o contêiner entra em **Error/Unhealthy**, e o Compose encerra com falha de dependência. **Não há evidência, neste log, de conflito de porta, erro de build ou falha do MinIO** — o gatilho é exclusivamente o **estado unhealthy do MLflow**.

# Linha do tempo (resumo factual)

1. `make up` inicia e executa builds (27/27 etapas da API; 5/5 do dashboard) → **sucesso**.
2. `docker compose up -d` sobe serviços:

   * `trustshield-minio-1`: **Healthy**,
   * `trustshield-minio-init-1`: **Exited**,
   * `trustshield-mlflow-1`: **Error** após \~134 s,
   * `trustshield-trustshield-api-1`: **Recreated**,
   * `trustshield-trustshield-dashboard-1`: **Created**.
3. Compose aborta: **“dependency failed to start: container trustshield-mlflow-1 is unhealthy”** → `make` encerra com erro.

> Em suma, **o problema é a não-saúde do contêiner MLflow durante a subida**, que aciona a política de dependências do Compose e derruba o `make up`. Para determinar a **causa técnica profunda** (ex.: parâmetros do servidor MLflow, backend store/DB inacessível, configuração S3/MinIO, entrypoint, porta, migrações), é necessário inspecionar os **logs do contêiner `trustshield-mlflow-1`** e o **healthcheck** definido para ele.
