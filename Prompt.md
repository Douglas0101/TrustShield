make check-env        # .env presente e sem variáveis aninhadas
make check-secrets    # confere secrets/ obrigatórios
make doctor           # checagem completa: Docker/Compose/curl/.env/secrets/portas
make config           # imprime o docker compose já resolvido
