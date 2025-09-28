# **TrustShield: Sistema Avançado de Detecção e Prevenção de Fraudes**

![CI/CD](https://img.shields.io/badge/CI%2FCd-passing-green?style=for-the-badge&logo=githubactions)
![Docker](https://img.shields.io/badge/Docker-ready-blue?style=for-the-badge&logo=docker)
![MLflow](https://img.shields.io/badge/MLflow-enabled-orange?style=for-the-badge&logo=m)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Licença](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Versão Empresarial: 5.1.0-stable**

O **TrustShield** é uma plataforma completa de *Data Science* e MLOps projetada para a detecção e prevenção de fraudes em transações financeiras em tempo real. Este projeto evolui de uma abordagem reativa para um sistema proativo e inteligente, utilizando modelos de *Machine Learning* não supervisionados para identificar padrões anómalos com alta precisão e eficiência.

O sistema foi concebido com uma arquitetura de nível empresarial, focada em robustez, escalabilidade e manutenibilidade, seguindo os princípios do *Domain-Driven Design* (DDD) e as melhores práticas de MLOps.

## **Índice**

- [Visão Geral e Objetivos](#visão-geral-e-objetivos)
- [Arquitetura do Projeto](#arquitetura-do-projeto)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Estrutura de Diretórios](#estrutura-de-diretórios)
- [Como Executar o Projeto](#como-executar-o-projeto)
  - [Pré-requisitos](#pré-requisitos)
  - [Configuração do Ambiente](#configuração-do-ambiente)
  - [Executando o Pipeline Completo](#executando-o-pipeline-completo)
- [Fluxo do Pipeline de Dados e ML](#fluxo-do-pipeline-de-dados-e-ml)
- [Qualidade de Código e Testes](#qualidade-de-código-e-testes)
- [Métricas de Sucesso e SLAs](#métricas-de-sucesso-e-slas)
- [Roadmap e Próximos Passos](#roadmap-e-próximos-passos)
- [Como Contribuir](#como-contribuir)
- [Licença](#licença)

---

## **Visão Geral e Objetivos**

O objetivo principal do TrustShield é minimizar perdas financeiras e fortalecer a confiança dos clientes através de um sistema de IA que aprende e se adapta continuamente.

-   **Deteção de Fraudes**: Identificar transações atípicas em tempo real para prevenção imediata.
-   **Aprendizagem Não Supervisionada**: Utilizar o modelo **Isolation Forest** como campeão para detecção de anomalias sem a necessidade de dados rotulados.
-   **Governança de Modelos**: Empregar **MLflow** para rastreamento de experimentos, versionamento de modelos e garantia de reprodutibilidade.
-   **Engenharia de Software Robusta**: Construir um sistema baseado em microserviços, conteinerizado com **Docker** e orquestrado para produção.

## **Arquitetura do Projeto**

O TrustShield adota uma **Arquitetura Hexagonal (Portas e Adaptadores)**, separando claramente o domínio de negócio das camadas de aplicação e infraestrutura. A comunicação entre os serviços é orientada a eventos, garantindo desacoplamento e escalabilidade.

-   **Conteinerização**: Todo o ambiente (serviços, banco de dados, armazenamento) é conteinerizado com **Docker** e orquestrado via **Docker Compose**.
-   **MLOps Stack**:
    -   **MLflow**: Para rastreamento de experimentos, registro de modelos e governança.
    -   **MinIO**: Como *storage* de objetos S3-compatível para armazenar os artefactos dos modelos.
    -   **PostgreSQL**: Como *backend* de armazenamento para o MLflow.
-   **Hardware Target**: Otimizado para execução em CPUs **Intel Core i3**, utilizando paralelismo e otimizações de memória.

## **Tecnologias Utilizadas**

-   **Linguagem**: Python 3.11+
-   **Bibliotecas de Dados**: Pandas, NumPy, Scikit-learn, PyArrow, Dask
-   **MLOps e Orquestração**: Docker, Docker Compose, MLflow, MinIO, PostgreSQL
-   **API**: FastAPI, Uvicorn
-   **Testes**: Pytest, Pytest-BDD, Hypothesis
-   **Gestão de Tarefas**: Makefile
-   **Configuração**: YAML

## **Estrutura de Diretórios**

A estrutura de diretórios segue as melhores práticas para projetos de *Data Science*, separando dados, código-fonte, configurações e saídas.

```
TrustShield/
├── README.md               # Visão geral do projeto, instruções de setup
├── LICENSE                 # Licença do projeto
├── config/                 # Arquivos de configuração (config.yaml)
├── data/
│   ├── raw/                # Dados brutos originais (CSV, JSON)
│   ├── interim/            # Dados pré-processados em etapas intermediárias
│   ├── processed/          # Dados finais prontos para treino/avaliação
│   └── external/           # Dados de fontes externas
├── docker/                 # Configuração do ambiente Docker
│   ├── Dockerfile
│   └── docker-compose.yml
├── notebooks/              # Jupyter Notebooks de exploração e prototipagem
├── outputs/                # Artefatos de saída (modelos, relatórios, gráficos)
│   ├── models/             # Modelos treinados (pickle, joblib)
│   └── figures/            # Gráficos de EDA e métricas
├── src/                    # Código-fonte do projeto
│   ├── api/                # Código da API de inferência
│   ├── data/               # Scripts para processamento de dados
│   ├── features/           # Scripts para engenharia de features
│   └── models/             # Scripts para treino, avaliação e predição de modelos
├── tests/                  # Testes unitários e de integração
├── Makefile                # Comandos para automatizar tarefas comuns
└── requirements.txt        # Dependências do Python
```

## **Como Executar o Projeto**

### **Pré-requisitos**

-   [Docker](https://docs.docker.com/get-docker/)
-   [Docker Compose](https://docs.docker.com/compose/install/)
-   `make` (geralmente já instalado em sistemas Linux/macOS)

### **Configuração do Ambiente**

1.  **Clone o repositório:**
    ```bash
    git clone [URL-DO-SEU-REPOSITÓRIO]
    cd TrustShield
    ```

2.  **Prepare os dados brutos:**
    Certifique-se de que os ficheiros de dados (`cards_data.csv`, `users_data.csv`, etc.) estão localizados no diretório `data/raw/`.

3.  **Acesso local à API:**
    A aplicação aplica uma _whitelist_ de IPs por padrão. O `docker-compose` já define `TRUSTSHIELD_ALLOWED_IPS` com a lista de redes privadas comuns (127.0.0.1, 10/8, 172.16/12, 192.168/16). Caso execute a API fora do Compose, ajuste esta variável para incluir os IPs autorizados ou defina `TRUSTSHIELD_DISABLE_IP_WHITELIST=true` para desativar a restrição durante o desenvolvimento.

4.  **Configuração de API Key:**
    Além da whitelist, os endpoints agora aceitam uma chave de API enviada no cabeçalho `X-API-Key` (ou no parâmetro `api_key`).
    -   Configure a variável `TRUSTSHIELD_API_KEYS` com uma lista de chaves separadas por vírgula. É possível fornecer valores em texto simples ou no formato `sha256:<hash>`. Alternativamente, utilize `TRUSTSHIELD_API_KEYS_FILE` apontando para um ficheiro seguro.
    -   O dashboard lê a chave a partir de `TRUSTSHIELD_API_KEY` (ou `TRUSTSHIELD_API_KEY_FILE`).
    -   Em ambientes de desenvolvimento o `docker-compose` define automaticamente a chave `local-dev-key`. Para produção, gere uma chave forte e actualize ambas as variáveis.

### **Executando o Pipeline Completo**

O `Makefile` simplifica a execução do projeto com os seguintes comandos:

1.  **Subir o Ambiente Completo:**
    Este comando constrói as imagens, sobe todos os serviços (API, Dashboard, MLflow, MinIO) e aguarda a API ficar saudável.
    ```bash
    make up
    ```

2.  **Executar o Pipeline de Treino (Ciclo Completo):**
    Após o ambiente estar no ar, este comando executa todas as etapas do pipeline de MLOps: processamento de dados, engenharia de features, treino, avaliação, otimização, validação, interpretação e promoção do modelo.
    ```bash
    make cycle
    ```
    *Para executar uma etapa individualmente (ex: `make train`), consulte o `Makefile`.*

3.  **Acompanhar os Logs:**
    Para visualizar os logs de todos os serviços em tempo real:
    ```bash
    make logs
    ```
    Para logs de um serviço específico (ex: API):
    ```bash
    make api-logs
    ```

4.  **Aceder aos Serviços:**
    -   **API (Health Check)**: [http://localhost:8080/healthz](http://localhost:8080/healthz)
    -   **Dashboard**: [http://localhost:8501](http://localhost:8501)
    -   **MLflow**: [http://localhost:5500](http://localhost:5500)

5.  **Parar o Ambiente:**
    Este comando para e remove todos os contêineres e redes.
    ```bash
    make down
    ```

6.  **Limpeza Completa:**
    Para uma limpeza mais profunda, removendo também os volumes, utilize:
    ```bash
    make clean
    ```
    Para limpar todo o cache do Docker (use com cuidado):
    ```bash
    make prune
    ```

## **Fluxo do Pipeline de Dados e ML**

O pipeline de MLOps é orquestrado pelo `Makefile` e executado dentro do contêiner de serviço da API (`trustshield-api`). O comando `make cycle` dispara a sequência completa:

1.  **`make data`**: Executa `src/data/make_dataset.py` para carregar os dados brutos de `data/raw`, limpá-los e criar um *dataset* primário em `data/processed`.
2.  **`make features`**: Executa `src/features/build_features.py` para aplicar engenharia de *features* e salvar o *dataset* final em `data/features`.
3.  **`make train`**: Executa `src/models/train_fraud_model.py` para treinar o modelo campeão (**Isolation Forest**), conectando-se ao MLflow para registrar parâmetros, métricas e o artefato do modelo.
4.  **`make eval`**: Avalia o modelo treinado.
5.  **`make optimize`**: Otimiza os hiperparâmetros do modelo.
6.  **`make promote`**: Promove o melhor modelo para produção.
7.  **`make reload`**: Reinicia o serviço da API para carregar o novo modelo.
8.  **`make smoke`**: Realiza um teste rápido para garantir que a API está funcionando corretamente.

## **Qualidade de Código e Testes**

-   **Convenções de Código**: PEP 8 para Python, Arquitetura Hexagonal e Domain-Driven Design (DDD).
-   **Revisão de Código**: Pull Requests obrigatórios com checks automatizados (lint, análise estática).
-   **Tipos de Testes**:
    -   **Unitários**: Pytest com mocking de conexões.
    -   **Integração**: Testes end-to-end com contêineres.
    -   **Avançados**: Testes de comportamento (BDD), baseados em propriedades e de contrato.

## **Métricas de Sucesso e SLAs**

-   **SLA de Inferência:** < 200ms p/ request em 95º percentil.
-   **Taxa de Detecção:** Recall ≥ 90% e Precision ≥ 85%.
-   **Disponibilidade:** ≥ 99.9% dos serviços principais.
-   **Taxa de Alerta Falso Positivo:** ≤ 2%.

## **Roadmap e Próximos Passos**

-   [ ] **API de Inferência**: Desenvolver uma API RESTful (com FastAPI) para servir o modelo campeão e realizar predições em tempo real.
-   [ ] **Dashboard de Monitoramento**: Criar um *dashboard* (com Streamlit ou Dash) para visualizar as predições e monitorar a saúde do modelo.
-   [ ] **Testes Automatizados**: Expandir a suíte de testes para incluir testes de integração para o pipeline completo.
-   [ ] **Deploy em Cloud**: Adaptar a configuração para *deploy* em um provedor de nuvem (AWS, GCP, Azure) utilizando Kubernetes.

## **Como Contribuir**

Contribuições são bem-vindas! Por favor, siga os seguintes passos:

1.  Faça um *fork* do projeto.
2.  Crie uma nova *branch* (`git checkout -b feature/sua-feature`).
3.  Faça o *commit* das suas alterações (`git commit -m 'Adiciona nova feature'`).
4.  Faça o *push* para a *branch* (`git push origin feature/sua-feature`).
5.  Abra um *Pull Request*.

## **Licença**

Este projeto está licenciado sob a Licença MIT. Veja o ficheiro `LICENSE` para mais detalhes.
