# TrustShield — Guia de Comandos (Makefile + Docker Compose)

> **Premissas**  
> • Makefile atualizado (usa: `docker compose -p trustshield -f docker/docker-compose.yml ...`)  
> • `docker/docker-compose.yml` está na versão otimizada (rede `trustshield`, secrets e `depends_on` robustos).  
> • Secrets presentes em `./secrets/`: `postgres_password.txt`, `minio_root_user.txt`, `minio_root_password.txt`.

---

## 1) Orquestração (use SEMPRE `make`)

Subir tudo (build + ordem correta):
```bash
make up
```

Ver estado dos serviços:
```bash
make ps
```

Logs (serviço específico):
```bash
make logs service=api
make logs service=mlflow
make logs service=postgres
make logs service=minio
make logs service=dashboard
```

Reiniciar a stack (rápido):
```bash
make restart
```

Derrubar a stack (containers, volumes do compose e órfãos):
```bash
make down
```

Limpeza radical (containers, imagens locais, volumes, redes do projeto):
```bash
make nuke    # pede confirmação (y/N)
```

---

## 2) Alternativa direta (Docker Compose "cru")

> Útil se você estiver depurando fora do `make`.

```bash
# Subir
docker compose -p trustshield -f docker/docker-compose.yml up -d --build

# Ver/seguir logs
docker compose -p trustshield -f docker/docker-compose.yml logs -f api

# Derrubar (completo, alinhado ao Makefile)
docker compose -p trustshield -f docker/docker-compose.yml down --volumes --remove-orphans

# Rebuild sem cache (quando trocar Dockerfiles)
docker compose -p trustshield -f docker/docker-compose.yml build --no-cache

# Validar o arquivo e variáveis
docker compose -p trustshield -f docker/docker-compose.yml config
```

---

## 3) Pipelines (ponta a ponta e granular)

Padrão (data → features → train → evaluate → promote):
```bash
make pipeline
```

Retreinamento com HPO (Optuna):
```bash
make retrain hpo_trials=50
```

Etapas isoladas:
```bash
make data
make features
make train experiment="Exp01" tag="ajuste-X"
make optimize hpo_trials=100
make evaluate
make promote
```

---

## 4) Verificações rápidas

API (saúde e status):
```bash
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:8000/status | jq .
```

Predição de exemplo:
```bash
curl -s -X POST http://127.0.0.1:8000/predict   -H "Content-Type: application/json"   -d '{"amount":250,"use_chip":"Chip","current_age":40,"retirement_age":65,
       "birth_year":1984,"gender":"M","latitude":34.05,"longitude":-118.25,
       "yearly_income":75000,"total_debt":15000,"credit_score":720,
       "num_credit_cards":4,"transaction_hour":15,"day_of_week":3,
       "is_weekend":false,"is_night_transaction":false,"amount_vs_avg":2.5}'
```

UIs:
```text
MLflow ......... http://127.0.0.1:5000
Dashboard ...... http://127.0.0.1:8501
MinIO Console .. http://127.0.0.1:9001
```

Secrets em uso (consulta local):
```bash
echo "MINIO_USER=$(cat secrets/minio_root_user.txt)"
echo "MINIO_PASS=$(cat secrets/minio_root_password.txt)"
```

---

## 5) Dicas & Troubleshooting

- **Aviso "Network ... No resource found to remove"** ao limpar? Normal quando a rede já não existe. Se quiser forçar:
  ```bash
  docker network rm trustshield trustshield_default docker_trustshield-net 2>/dev/null || true
  ```

- **Porta ocupada** (8000/8501/5000/9000/9001)? Pare serviços concorrentes ou altere o mapeamento de portas no compose.

- **MLflow não responde na API**: garanta que `MLFLOW_TRACKING_URI=http://mlflow:5000` está no serviço `api` e que o `entrypoint.sh` foi copiado na imagem (o `Makefile`/Dockerfile já cuidam disso).

- **MinIO**: o job `minio-init` cria o bucket `mlflow` e aplica política privada. Ajuste política no compose se precisar público.

---

## 6) Adoção da versão otimizada do Compose

Se você gerou um `docker-compose.optimized.yml`, torne-o o padrão do projeto:
```bash
mv docker/docker-compose.optimized.yml docker/docker-compose.yml
# (ou atualize a variável COMPOSE_FILE no Makefile)
```
