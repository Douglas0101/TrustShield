# TrustShield: Projeto de Detecção e Prevenção de Fraudes em Tempo Real

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.9-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## 📄 Resumo Executivo

O TrustShield é uma solução de Data Science e Inteligência Artificial projetada para identificar padrões atípicos em transações financeiras em tempo real. O objetivo é reduzir substancialmente as perdas por fraudes, fortalecer a confiança dos clientes e melhorar a eficiência operacional através de um mecanismo de detecção proativo e inteligente.

---

## 🎯 Contexto: O Desafio dos Dados Não Rotulados

O objetivo inicial do projeto era treinar um modelo de aprendizado supervisionado. Contudo, uma análise exploratória revelou que o conjunto de dados de treinamento não continha nenhum exemplo de fraude rotulada (`label=1`). Este desafio exigiu uma mudança estratégica na abordagem de modelagem.

## 💡 Solução: Detecção de Anomalias com Isolation Forest

Para superar a ausência de rótulos, o projeto pivotou com sucesso para uma abordagem de aprendizado não supervisionado, utilizando a técnica de **Detecção de Anomalias**. O algoritmo `Isolation Forest` foi selecionado como campeão devido à sua eficácia em grandes volumes de dados e sua capacidade comprovada de identificar transações suspeitas, ao contrário de outros modelos como One-Class SVM e LOF que não detectaram anomalias neste dataset.

### 📈 Perfil da Anomalia

A análise das anomalias detectadas pelo `Isolation Forest` revelou um perfil claro e consistente:

* **Valor da Transação:** As transações anômalas possuem valores drasticamente superiores (média de **$418,73**) em comparação com as normais (média de **$42,60**).
* **Horário da Transação:** Apresentam picos de atividade em horários não convencionais, especificamente de madrugada (entre 2h e 5h) e no final da noite.
* **Perfil do Cliente:** Estão associadas a clientes com alta renda anual (média de **$123k** vs. $46k) mas com score de crédito inferior (média de **665** vs. 714).

---

## 🛠️ Arquitetura e Estrutura do Projeto

O repositório segue uma estrutura modular para garantir clareza, manutenibilidade e reprodutibilidade:

TrustShield/
├── config/                 # Arquivos de configuração (YAML)

├── data/                   # Dados brutos, intermediários e processados

├── notebooks/              # Notebooks para exploração (EDA)

├── outputs/                # Artefatos gerados (modelos, relatórios, logs)

├── src/                    # Código-fonte principal

│   ├── data/               # Scripts para criação do dataset

│   ├── features/           # Scripts para engenharia de features

│   └── models/             # Scripts de treinamento e inferência

├── tests/                  # Testes unitários e de integração

├── scripts/                # Scripts de automação (Makefile)

├── Dockerfile              # Definição do container da aplicação

└── docker-compose.yml      # Orquestração de serviços (ex: MLflow)

---

## 🚀 Tecnologias Utilizadas

* **Core ML:** Scikit-learn, Pandas, NumPy
* **Experiment Tracking:** MLflow (planejado)
* **Armazenamento de Dados:** Parquet
* **Containerização:** Docker
* **Automação:** Makefile
* **Qualidade de Código:** Black, Flake8, MyPy
* **Testes:** Pytest

---

## ⚙️ Como Executar o Projeto

### 1. Pré-requisitos
- Python 3.9+
- Docker e Docker Compose (para ambiente containerizado)

### 2. Instalação de Dependências
Use o `Makefile` para instalar todas as dependências necessárias:
```bash

make install
