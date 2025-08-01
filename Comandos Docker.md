# 1. Limpeza total (opcional, mas recomendado para um começo limpo)
make purge

# 2. Remover todas as imagens Docker não utilizadas (opcional, mas recomendado):
docker image prune -a -f

# 3. Iniciar os serviços de suporte
docker compose up --build mlflow
make services-up-fresh

# 4. Executar o treino
make train
