# **TrustShield: Sistema Avançado de Detecção e Prevenção de Fraudes**

![CI/CD](https://img.shields.io/badge/CI%2FCd-passing-green?style=for-the-badge&logo=githubactions)
![Docker](https://img.shields.io/badge/Docker-ready-blue?style=for-the-badge&logo=docker)
![MLflow](https://img.shields.io/badge/MLflow-enabled-orange?style=for-the-badge&logo=m)
![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
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

-   **Linguagem**: Python 3.10+
-   **Bibliotecas de Dados**: Pandas, NumPy, Scikit-learn, PyArrow
-   **MLOps e Orquestração**: Docker, Docker Compose, MLflow, MinIO, PostgreSQL
-   **Gestão de Tarefas**: Makefile
-   **Configuração**: YAML

## **Estrutura de Diretórios**

A estrutura de diretórios segue as melhores práticas para projetos de *Data Science*, separando dados, código-fonte, configurações e saídas.

````
├── config/                # Ficheiros de configuração (config.yaml)
├── data/                  # Dados do projeto (brutos, processados, features)
│   ├── raw/
│   ├── processed/
│   └── features/
├── docker/                # Configuração do ambiente Docker
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/                  # Documentação do projeto (PDFs, Markdown)
├── notebooks/             # Jupyter Notebooks para análise exploratória (EDA)
├── outputs/               # Saídas geradas (modelos, relatórios, figuras)
│   ├── models/
│   └── reports/
├── src/                   # Código-fonte do projeto
│   ├── data/              # Scripts para processamento de dados (make_dataset.py)
│   ├── features/          # Scripts para engenharia de features (build_features.py)
│   └── models/            # Scripts para treino (train_model.py) e inferência (predict.py)
├── tests/                 # Testes unitários e de integração
├── Makefile               # Comandos para automatizar tarefas comuns
├── requirements.txt       # Dependências do Python
└── README.md              # Este ficheiro
````

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

### **Executando o Pipeline Completo**

O `Makefile` automatiza todo o processo. Os comandos devem ser executados na raiz do projeto.

1.  **Limpeza Total (Recomendado para a primeira execução ou após alterações):**
    Este comando para e remove todos os contentores, volumes e redes, além de limpar o cache do Docker para evitar conflitos.
    ```bash
    make docker-stop && docker system prune -a -f
    ```

2.  **Construir a Imagem Docker:**
    Este comando constrói a imagem principal com todas as dependências do projeto.
    ```bash
    make docker-build
    ```

3.  **Executar o Ambiente e o Pipeline:**
    Este comando sobe todos os serviços (PostgreSQL, MinIO, MLflow) e inicia o contentor de treino, que executará o pipeline completo de dados e ML.
    ```bash
    make docker-run
    ```

4.  **Acompanhar os Logs:**
    Para ver o progresso do pipeline em tempo real, use este comando:
    ```bash
    make docker-logs
    ```

5.  **Aceder à Interface do MLflow:**
    Abra o seu navegador e aceda a **[http://localhost:5000](http://localhost:5000)** para ver os experimentos, execuções e modelos registados.

6.  **Parar o Ambiente:**
    Quando terminar, use este comando para parar e remover todos os contentores e volumes.
    ```bash
    make docker-stop
    ```

## **Fluxo do Pipeline de Dados e ML**

Quando `make docker-run` é executado, o seguinte pipeline é orquestrado dentro do contentor `trustshield-trainer`:

1.  **`src/data/make_dataset.py`**: Carrega os dados brutos de `data/raw`, limpa-os e cria um *dataset* primário em `data/processed`.
2.  **`src/features/build_features.py`**: Aplica engenharia de *features* sobre o *dataset* primário e salva o *dataset* final em `data/features`.
3.  **`src/models/train_fraud_model.py`**:
    -   Conecta-se ao servidor MLflow.
    -   Carrega o *dataset* de *features*.
    -   Treina o modelo campeão (**Isolation Forest**).
    -   Regista parâmetros, métricas e o artefacto do modelo no MLflow.

## **Roadmap e Próximos Passos**

-   [ ] **API de Inferência**: Desenvolver uma API RESTful (ex: com FastAPI) para servir o modelo campeão e realizar predições em tempo real.
-   [ ] **Dashboard de Monitoramento**: Criar um *dashboard* (ex: com Streamlit ou Dash) para visualizar as predições e monitorar a saúde do modelo.
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