A ferramenta de execução de código não permite a execução de comandos de shell do sistema operacional como `sudo lsof`. **Você deve executar esse comando no seu próprio terminal do Ubuntu** para ver qual processo está usando a porta 9000.

Independentemente do processo encontrado, a solução permanente e recomendada continua sendo a mesma. Prossiga com os passos seguintes.

---
**Passo 2: Modificação Estratégica da Porta no Docker Compose**
Esta é a correção principal. Você irá alterar a porta que o serviço `minio` expõe na sua máquina (host), sem alterar a porta interna do contêiner.

1.  **Abra o arquivo de configuração:**
    ```bash
    gedit docker/docker-compose.yml
    ```

2.  **Localize e altere a seção `ports` do serviço `minio`:**

    **Encontre este bloco:**
    ```yaml
    services:
      minio:
        # ... outras chaves de configuração ...
        ports:
          - "9000:9000"  # <-- LINHA DO CONFLITO
          - "9001:9001"
    ```

    **Altere-o para o seguinte:**
    ```yaml
    services:
      minio:
        # ... outras chaves de configuração ...
        ports:
          - "9900:9000"  # <-- CORREÇÃO APLICADA
          - "9901:9001"  # Recomendado alterar a porta da console também
    ```
    *   **Justificativa Técnica:** Esta alteração mapeia a porta `9900` (livre) do seu host para a porta `9000` interna do contêiner `minio`. A comunicação entre os outros contêineres (ex: `api` -> `minio`) não é afetada, pois ocorre na rede interna do Docker, que continuará usando `minio:9000`.

**Passo 3: Re-execução e Validação do Ambiente**
Após salvar o arquivo `docker/docker-compose.yml` modificado, execute novamente o comando para subir a stack.

1.  **Execute o `make up`:**
    ```bash
    make up
    ```
    O comando agora deve ser concluído com sucesso, sem o erro de alocação de porta.

2.  **Valide o acesso:**
    Para confirmar que o serviço está no ar, acesse a interface do MinIO no seu navegador utilizando a **nova porta**:
    `http://localhost:9900`