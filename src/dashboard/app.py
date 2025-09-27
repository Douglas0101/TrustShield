# -*- coding: utf-8 -*-
"""
TrustShield Dashboard (Streamlit) — versão otimizada e completa
- API_URL agora lê a variável de ambiente, com fallback para o nome do serviço Docker.
- Leitura paginada de Parquet usando pyarrow.dataset (com filtros).
- Limites de exibição para evitar travamentos.
- Threads BLAS limitadas no topo.
- Nada de polling agressivo; ações dirigidas por botões.

Como executar (a partir da raiz do projeto):
    # Dentro do Docker, já é executado automaticamente.
    # Localmente (para desenvolvimento da UI, se a API estiver rodando via 'make up'):
    streamlit run src/dashboard/app.py
"""

# -----------------------------------------------------------------------------
# Limitar threads de BLAS/Numba ANTES dos imports pesados
# -----------------------------------------------------------------------------
import os as _os

for _v in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
]:
    _os.environ.setdefault(_v, "1")

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import json
import os
from pathlib import Path

import requests
import yaml
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import pyarrow.dataset as ds

import config_path  # noqa: F401  # garante que src esteja no sys.path em execuções locais
from src.common.paths import repo_root

# -----------------------------------------------------------------------------
# Configurações básicas
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TrustShield Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ TrustShield - Dashboard de Monitoramento de Fraudes")
st.caption(
    "Painel otimizado para análise operacional, investigativa e de performance do modelo."
)

# -----------------------------------------------------------------------------
# Config e limites
# -----------------------------------------------------------------------------
ROOT = repo_root()
CONFIG_PATH = ROOT / "config" / "config.yaml"

CONFIG: dict = {}
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        CONFIG = yaml.safe_load(fh) or {}
except FileNotFoundError:
    st.warning(
        "Ficheiro de configuração não encontrado; a usar caminhos padrão por defeito."
    )

PATHS_SECTION = CONFIG.get("paths", {}) if isinstance(CONFIG, dict) else {}
DATA_PATHS = PATHS_SECTION.get("data", {})
FEATURED_DATASET_REL = DATA_PATHS.get(
    "featured_dataset", "data/features/featured_dataset.parquet"
)
FEATURED_DATASET_PATH = ROOT / FEATURED_DATASET_REL

OUTPUT_PATHS = PATHS_SECTION.get("outputs", {})
VALIDATION_PATHS = OUTPUT_PATHS.get("validation", {}) if isinstance(OUTPUT_PATHS, dict) else {}
DRIFT_REPORT_REL = (
    VALIDATION_PATHS.get("drift_detection")
    if isinstance(VALIDATION_PATHS, dict)
    else None
)
if not DRIFT_REPORT_REL:
    DRIFT_REPORT_REL = "outputs/validation/drift_detection"
DRIFT_REPORT_DIR = ROOT / DRIFT_REPORT_REL

API_URL = os.getenv("TRUSTSHIELD_API_URL")
if not API_URL:
    run_in_docker = os.getenv("RUN_IN_DOCKER", "").lower() in {"1", "true", "yes"}
    API_URL = "http://trustshield-api:8000" if run_in_docker else "http://localhost:8000"

SAMPLE_PREDICTION_PAYLOAD = {
    "amount": 120.5,
    "gender": "Male",
    "use_chip": "Swipe Transaction",
    "credit_score": 720,
    "num_credit_cards": 3,
    "per_capita_income": 23679,
    "yearly_income": 48277,
    "total_debt": 110153,
}

MAX_FEED_ROWS = 2000  # máximo de linhas no feed em memória
MAX_MAP_POINTS = 500  # máximo de pontos no mapa
PLOT_SAMPLE = 5000  # amostra máx para gráficos
PAGE_SIZE = 1000  # paginação real do parquet

if "anomaly_feed" not in st.session_state:
    st.session_state.anomaly_feed = pd.DataFrame(
        columns=["Timestamp", "Label", "Score", "Amount", "Latitude", "Longitude"]
    )

if "page_num" not in st.session_state:
    st.session_state.page_num = 1


# -----------------------------------------------------------------------------
# Funções utilitárias (API)
# -----------------------------------------------------------------------------
def get_api_status():
    try:
        r = requests.get(f"{API_URL}/status", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"status": "UNAVAILABLE", "error": str(e)}


def predict_transaction(transaction_data: dict):
    try:
        response = requests.post(
            f"{API_URL}/predict", json=transaction_data, timeout=10
        )
    except requests.exceptions.RequestException as exc:
        return {
            "success": False,
            "status_code": None,
            "error": f"Não foi possível contactar a API: {exc}",
        }

    try:
        payload = response.json()
    except ValueError:
        payload = {"detail": response.text}

    if response.status_code == 200:
        is_anomaly_raw = payload.get("is_anomaly")
        is_anomaly = bool(is_anomaly_raw)
        if isinstance(is_anomaly_raw, (int, float)):
            is_anomaly = int(is_anomaly_raw) == -1
        elif isinstance(is_anomaly_raw, str):
            is_anomaly = is_anomaly_raw.strip().lower() in {"-1", "true", "anomaly"}

        label = "ANOMALIA" if is_anomaly else "NORMAL"
        return {
            "success": True,
            "status_code": response.status_code,
            "data": payload,
            "prediction_label": label,
            "is_anomaly": is_anomaly,
            "confidence_score": payload.get("score"),
            "model_version": payload.get("model_version"),
        }

    detail = payload.get("detail", "Resposta inesperada da API.")
    if isinstance(detail, list):
        formatted = []
        for item in detail:
            if isinstance(item, dict):
                formatted.append(item.get("msg") or item.get("detail") or str(item))
            else:
                formatted.append(str(item))
        detail = "; ".join(formatted)
    elif isinstance(detail, dict):
        detail = detail.get("msg") or detail.get("error") or str(detail)

    return {
        "success": False,
        "status_code": response.status_code,
        "error": detail,
        "detail": payload,
    }


def explain_transaction(transaction_data: dict):
    try:
        r = requests.post(f"{API_URL}/explain", json=transaction_data, timeout=10)
        if r.status_code == 202:
            return {"success": True, "explanation": r.json()}
        r.raise_for_status()
        return {"success": True, "explanation": r.json()}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}


def get_explanation_result(job_id: str):
    try:
        r = requests.get(f"{API_URL}/explanation-result/{job_id}", timeout=5)
        if r.status_code == 202:
            return {"success": False, "status_code": 202, "error": "Result not ready"}
        r.raise_for_status()
        return {"success": True, "explanation": r.json()}
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e),
            "status_code": getattr(getattr(e, "response", None), "status_code", 500),
        }


def validate_model():
    try:
        r = requests.post(f"{API_URL}/validate", timeout=10)
        r.raise_for_status()
        return {"success": True, "data": r.json()}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}


# -----------------------------------------------------------------------------
# Dados: leitura paginada via pyarrow.dataset
# -----------------------------------------------------------------------------
def fetch_filtered_data_streaming(
    parquet_path: Path | str,
    amount_range,
    income_range,
    page_num: int = 1,
    page_size: int = PAGE_SIZE,
):
    """Leitura realmente paginada via pyarrow.dataset: filtra no leitor e só traz a página pedida."""
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        st.warning(
            "Dataset de features não encontrado. Gere-o através do pipeline antes de continuar."
        )
        return pd.DataFrame(), 0

    try:
        dataset = ds.dataset(str(parquet_path), format="parquet")
    except Exception as e:
        st.error(f"Falha ao abrir dataset Parquet: {e}")
        return pd.DataFrame(), 0

    columns = [
        "amount",
        "yearly_income",
        "transaction_hour",
        "day_of_week",
        "is_weekend",
        "is_night_transaction",
        "amount_vs_avg",
    ]

    filt = (
        (ds.field("amount") >= float(amount_range[0]))
        & (ds.field("amount") <= float(amount_range[1]))
        & (ds.field("yearly_income") >= float(income_range[0]))
        & (ds.field("yearly_income") <= float(income_range[1]))
    )

    # 1) Contar total
    total_records = 0
    for batch in dataset.to_batches(columns=columns, filter=filt, batch_size=10000):
        total_records += batch.num_rows

    # 2) Trazer apenas a página
    start_index = (page_num - 1) * page_size
    end_index = start_index + page_size

    collected = []
    seen = 0
    for batch in dataset.to_batches(columns=columns, filter=filt, batch_size=page_size):
        nb = batch.num_rows
        next_seen = seen + nb
        if next_seen <= start_index:
            seen = next_seen
            continue
        s = max(0, start_index - seen)
        e = min(nb, end_index - seen)
        if s < e:
            collected.append(batch.slice(s, e - s).to_pandas())
        seen = next_seen
        if seen >= end_index:
            break

    df_page = pd.concat(collected, ignore_index=True) if collected else pd.DataFrame()
    return df_page, total_records


# -----------------------------------------------------------------------------
# Visualização auxiliar
# -----------------------------------------------------------------------------
def create_waterfall_plot(explanation: dict):
    shap_values = explanation["shap_values"][0]
    base_value = explanation["base_values"][0]
    feature_names = explanation["feature_names"]

    fig = go.Figure(
        go.Waterfall(
            name="SHAP",
            orientation="v",
            measure=["relative"] * len(feature_names),
            x=feature_names,
            textposition="outside",
            text=[f"{val:.2f}" for val in shap_values],
            y=shap_values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            base=base_value,
        )
    )
    fig.update_layout(
        title="Análise de Contribuição das Features (SHAP)", showlegend=True
    )
    return fig


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Status da API")
    st.caption(f"Endpoint configurado: {API_URL}")
    if st.button("Atualizar Status", use_container_width=True):
        st.session_state.api_status = get_api_status()

    if "api_status" not in st.session_state:
        st.session_state.api_status = get_api_status()

    api_status = st.session_state.api_status
    if api_status.get("status") == "OPERATIONAL":
        st.success(f"**Status:** {api_status.get('status')}")
        st.write(f"**Modelo:** {api_status.get('model_type', 'N/A')}")
    else:
        st.error(f"**Status:** {api_status.get('status', 'ERRO')}")
        st.caption(api_status.get("error", "API indisponível."))

    st.header("Ações")
    st.write("Utilize a amostra abaixo para validar rapidamente o endpoint /predict.")
    st.code(
        json.dumps(SAMPLE_PREDICTION_PAYLOAD, indent=2, ensure_ascii=False),
        language="json",
    )
    if st.button("Testar Predição com Amostra", use_container_width=True):
        sample_data = dict(SAMPLE_PREDICTION_PAYLOAD)
        st.session_state.last_prediction_input = sample_data
        prediction = predict_transaction(sample_data)
        st.session_state.last_prediction_result = prediction
        st.session_state.last_explanation = None

        if prediction.get("success") and prediction.get("is_anomaly"):
            new_entry = pd.DataFrame(
                [
                    {
                        "Timestamp": pd.to_datetime(
                            prediction.get("data", {}).get(
                                "timestamp", pd.Timestamp.utcnow()
                            )
                        ),
                        "Label": prediction.get("prediction_label"),
                        "Score": prediction.get("confidence_score"),
                        "Amount": sample_data.get("amount"),
                        "Latitude": sample_data.get("latitude"),
                        "Longitude": sample_data.get("longitude"),
                    }
                ]
            )
            st.session_state.anomaly_feed = pd.concat(
                [new_entry, st.session_state.anomaly_feed], ignore_index=True
            )
            if len(st.session_state.anomaly_feed) > MAX_FEED_ROWS:
                st.session_state.anomaly_feed = st.session_state.anomaly_feed.head(
                    MAX_FEED_ROWS
                )

    st.header("Filtros de Análise Histórica")
    amount_range = st.slider(
        "Valor da Transação (Amount)",
        min_value=0.0,
        max_value=10000.0,
        value=(0.0, 1000.0),
    )

    income_range = st.slider(
        "Renda Anual (Yearly Income)",
        min_value=0.0,
        max_value=300000.0,
        value=(0.0, 100000.0),
    )

    if st.button("Aplicar filtros", use_container_width=True):
        st.session_state.page_num = 1

# -----------------------------------------------------------------------------
# Abas
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["Monitoramento em Tempo Real", "Análise Investigativa", "Performance do Modelo"]
)

# -----------------------------------------------------------------------------
# Tab 1 — Operacional
# -----------------------------------------------------------------------------
with tab1:
    st.header("Visão Operacional")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(
            label="Total de Anomalias (Sessão)",
            value=len(st.session_state.anomaly_feed),
        )
    with kpi2:
        st.metric(
            label="Valor Monetário das Anomalias",
            value=f"R$ {st.session_state.anomaly_feed['Amount'].sum():.2f}",
        )
    with kpi3:
        st.metric(label="Taxa de Anomalias", value="0.00%")
    with kpi4:
        st.metric(label="Latência Média da API", value="N/A")

    st.markdown("---")
    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("Feed de Anomalias ao Vivo")
        if not st.session_state.anomaly_feed.empty:
            st.dataframe(
                st.session_state.anomaly_feed.head(500), use_container_width=True
            )
        else:
            st.info("Sem anomalias na sessão até o momento.")

    with col2:
        st.subheader("Mapa de Atividade Suspeita")
        if not st.session_state.anomaly_feed.empty:
            st.map(
                st.session_state.anomaly_feed[["Latitude", "Longitude"]].tail(
                    MAX_MAP_POINTS
                )
            )
        else:
            st.info("Aguardando detecção de anomalias para exibir no mapa.")

    st.header("Deep Dive da Última Predição")
    if "last_prediction_result" in st.session_state:
        result = st.session_state.last_prediction_result
        if result.get("success"):
            c1, c2 = st.columns(2)
            with c1:
                st.success(f"**Resultado:** {result.get('prediction_label', 'N/A')}")
                st.caption(
                    f"Score: {result.get('confidence_score', 'N/A')} | Modelo: {result.get('model_version', 'N/A')}"
                )
                if st.button("Explicar Predição (SHAP)"):
                    with st.spinner("Iniciando análise de explicação..."):
                        explanation_job = explain_transaction(
                            st.session_state.last_prediction_input
                        )
                        if explanation_job.get("success"):
                            st.session_state.explanation_job_id = explanation_job.get(
                                "explanation", {}
                            ).get("job_id")
                            st.session_state.last_explanation = None
                            st.info(
                                "Análise de explicação iniciada. Verifique o resultado em breve."
                            )
                        else:
                            st.error(
                                f"Falha ao iniciar a explicação: {explanation_job.get('error')}"
                            )

            with c2:
                st.write("**Dados da Transação Enviada:**")
                st.json(st.session_state.last_prediction_input, expanded=False)

            # Botão manual para buscar o resultado (evita polling agressivo)
            if st.session_state.get("explanation_job_id") and not st.session_state.get(
                "last_explanation"
            ):
                if st.button("Verificar Resultado da Explicação"):
                    with st.spinner("Buscando resultado..."):
                        r = get_explanation_result(st.session_state.explanation_job_id)
                        if r.get("success"):
                            st.session_state.last_explanation = {
                                "success": True,
                                "explanation": r.get("explanation"),
                            }
                            st.session_state.explanation_job_id = None
                            st.rerun()
                        elif r.get("status_code") == 202:
                            st.info(
                                "A análise ainda está em andamento. Tente novamente em alguns segundos."
                            )
                        else:
                            st.error(f"Erro ao buscar resultado: {r.get('error')}")

            if st.session_state.get("last_explanation"):
                exp = st.session_state.last_explanation
                if exp.get("success"):
                    st.plotly_chart(
                        create_waterfall_plot(exp["explanation"]),
                        use_container_width=True,
                    )
                else:
                    st.error(f"Falha ao gerar explicação: {exp.get('error')}")
        else:
            error_message = result.get("error", "Falha desconhecida na predição.")
            if result.get("status_code") in {400, 422}:
                st.warning(
                    "Entrada inválida: "
                    + error_message
                    + " — valide os campos obrigatórios ou utilize 'Testar Predição com Amostra'."
                )
            else:
                st.error(
                    "Não foi possível concluir a predição: "
                    + error_message
                )
            with st.expander("Payload enviado", expanded=False):
                st.json(st.session_state.last_prediction_input, expanded=False)
            st.caption(
                "Sugestão: na barra lateral clique em 'Testar Predição com Amostra' para um exemplo válido."
            )
    else:
        st.info("Clique em 'Testar Predição com Amostra' para ver os detalhes.")

# -----------------------------------------------------------------------------
# Tab 2 — Investigativa (Histórico)
# -----------------------------------------------------------------------------
with tab2:
    st.header("Análise Tática de Dados Históricos")

    st.subheader("Explorador de Dados (paginado)")
    filtered_data, total_records = fetch_filtered_data_streaming(
        FEATURED_DATASET_PATH,
        amount_range,
        income_range,
        st.session_state.page_num,
        PAGE_SIZE,
    )

    if not filtered_data.empty:
        st.dataframe(filtered_data, use_container_width=True)
        st.info(
            f"Mostrando {len(filtered_data)} de {total_records} registros filtrados."
        )

        total_pages = (total_records // PAGE_SIZE) + int(total_records % PAGE_SIZE > 0)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            if st.button("Página Anterior") and st.session_state.page_num > 1:
                st.session_state.page_num -= 1
                st.rerun()
        with c3:
            if st.button("Próxima Página") and st.session_state.page_num < total_pages:
                st.session_state.page_num += 1
                st.rerun()
        with c2:
            st.write(f"Página {st.session_state.page_num} de {max(total_pages, 1)}")
    else:
        st.warning("Nenhum registro encontrado para os filtros aplicados.")

    st.markdown("---")
    st.subheader("Visualizações Interativas (baseadas na página atual)")
    if not filtered_data.empty:
        # amostra para gráficos
        sample = filtered_data.sample(
            min(PLOT_SAMPLE, len(filtered_data)), random_state=42
        )
        c1, c2 = st.columns(2)
        with c1:
            fig_hist = px.histogram(
                sample, x="transaction_hour", title="Distribuição por Hora (Amostra)"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        with c2:
            fig_box = px.box(
                sample, y="amount", title="Distribuição de Valores (Amostra)"
            )
            st.plotly_chart(fig_box, use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 3 — Performance do Modelo
# -----------------------------------------------------------------------------
with tab3:
    st.header("Visão Estratégica de Performance")

    if st.button("Gerar Relatório de Drift de Dados"):
        with st.spinner("Gerando relatório..."):
            result = validate_model()
            if result.get("success"):
                st.success("Validação iniciada com sucesso!")
            else:
                st.error(f"Falha ao iniciar validação: {result.get('error')}")

    st.markdown("---")
    st.subheader("Relatórios de Validação Disponíveis")

    report_dir = DRIFT_REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    report_files = sorted(p.name for p in report_dir.glob("*.html"))
    if report_files:
        selected_report = st.selectbox(
            "Selecione um relatório para visualizar", report_files
        )
        if selected_report:
            with open(report_dir / selected_report, "r", encoding="utf-8") as f:
                html_content = f.read()
            components.html(html_content, height=600, scrolling=True)
    else:
        st.info("Sem relatórios gerados ainda.")
