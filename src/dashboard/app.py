# -*- coding: utf-8 -*-
"""
TrustShield Dashboard (Streamlit) — versão otimizada e completa
- API_URL fixo em http://127.0.0.1:8000
- Leitura paginada de Parquet usando pyarrow.dataset (com filtros)
- Limites de exibição para evitar travamentos
- Threads BLAS limitadas no topo
- Nada de polling agressivo; ações dirigidas por botões

Como executar (a partir da raiz do projeto):
    cd ~/PycharmProjects/TrustShield
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    streamlit run src/dashboard/app.py \
      --server.headless=true \
      --server.fileWatcherType=none \
      --server.runOnSave=false \
      --logger.level=error
"""

# -----------------------------------------------------------------------------
# Limitar threads de BLAS/Numba ANTES dos imports pesados
# -----------------------------------------------------------------------------
import os as _os
for _v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"]:
    _os.environ.setdefault(_v, "1")

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import pyarrow.dataset as ds

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
st.caption("Painel otimizado para análise operacional, investigativa e de performance do modelo.")

# -----------------------------------------------------------------------------
# Config e limites
# -----------------------------------------------------------------------------
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
PARQUET_PATH = "data/features/featured_dataset.parquet"

MAX_FEED_ROWS = 2000   # máximo de linhas no feed em memória
MAX_MAP_POINTS = 500   # máximo de pontos no mapa
PLOT_SAMPLE = 5000     # amostra máx para gráficos
PAGE_SIZE = 1000       # paginação real do parquet

if "anomaly_feed" not in st.session_state:
    st.session_state.anomaly_feed = pd.DataFrame(columns=[
        "Timestamp", "Label", "Score", "Amount", "Latitude", "Longitude"
    ])

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
        r = requests.post(f"{API_URL}/predict", json=transaction_data, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}

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
        return {"success": False, "error": str(e), "status_code": getattr(getattr(e, 'response', None), 'status_code', 500)}

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
def fetch_filtered_data_streaming(parquet_path: str, amount_range, income_range, page_num=1, page_size=PAGE_SIZE):
    """Leitura realmente paginada via pyarrow.dataset: filtra no leitor e só traz a página pedida."""
    try:
        dataset = ds.dataset(parquet_path, format="parquet")
    except Exception as e:
        st.error(f"Falha ao abrir dataset Parquet: {e}")
        return pd.DataFrame(), 0

    columns = [
        'amount', 'yearly_income', 'transaction_hour', 'day_of_week',
        'is_weekend', 'is_night_transaction', 'amount_vs_avg'
    ]

    filt = (
        (ds.field('amount') >= float(amount_range[0])) &
        (ds.field('amount') <= float(amount_range[1])) &
        (ds.field('yearly_income') >= float(income_range[0])) &
        (ds.field('yearly_income') <= float(income_range[1]))
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
    shap_values = explanation['shap_values'][0]
    base_value = explanation['base_values'][0]
    feature_names = explanation['feature_names']

    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="v",
        measure=["relative"] * len(feature_names),
        x=feature_names,
        textposition="outside",
        text=[f"{val:.2f}" for val in shap_values],
        y=shap_values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        base=base_value,
    ))
    fig.update_layout(title="Análise de Contribuição das Features (SHAP)", showlegend=True)
    return fig

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Status da API")
    if st.button("Atualizar Status", use_container_width=True):
        st.session_state.api_status = get_api_status()

    if 'api_status' not in st.session_state:
        st.session_state.api_status = get_api_status()

    api_status = st.session_state.api_status
    if api_status.get("status") == "OPERATIONAL":
        st.success(f"**Status:** {api_status.get('status')}")
        st.write(f"**Modelo:** {api_status.get('model_type', 'N/A')}")
    else:
        st.error(f"**Status:** {api_status.get('status', 'ERRO')}")
        st.caption(api_status.get('error', 'API indisponível.'))

    st.header("Ações")
    if st.button("Testar Predição com Amostra", use_container_width=True):
        sample_data = {
            'amount': 250.0, 'use_chip': 'Chip', 'current_age': 40, 'retirement_age': 65,
            'birth_year': 1984, 'gender': 'M', 'latitude': 34.05, 'longitude': -118.25,
            'yearly_income': 75000, 'total_debt': 15000, 'credit_score': 720,
            'num_credit_cards': 4, 'transaction_hour': 15, 'day_of_week': 3,
            'is_weekend': False, 'is_night_transaction': False, 'amount_vs_avg': 2.5
        }
        st.session_state.last_prediction_input = sample_data
        prediction = predict_transaction(sample_data)
        st.session_state.last_prediction_result = prediction
        st.session_state.last_explanation = None

        if prediction.get("success") and prediction.get("prediction_label") == "ANOMALIA":
            new_entry = pd.DataFrame([{
                "Timestamp": pd.to_datetime(prediction.get("timestamp", pd.Timestamp.utcnow())),
                "Label": prediction.get("prediction_label"),
                "Score": prediction.get("confidence_score"),
                "Amount": sample_data.get("amount"),
                "Latitude": sample_data.get("latitude"),
                "Longitude": sample_data.get("longitude"),
            }])
            st.session_state.anomaly_feed = pd.concat([new_entry, st.session_state.anomaly_feed], ignore_index=True)
            # cap o tamanho do feed
            if len(st.session_state.anomaly_feed) > MAX_FEED_ROWS:
                st.session_state.anomaly_feed = st.session_state.anomaly_feed.head(MAX_FEED_ROWS)

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
tab1, tab2, tab3 = st.tabs(["Monitoramento em Tempo Real", "Análise Investigativa", "Performance do Modelo"])

# -----------------------------------------------------------------------------
# Tab 1 — Operacional
# -----------------------------------------------------------------------------
with tab1:
    st.header("Visão Operacional")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(label="Total de Anomalias (Sessão)", value=len(st.session_state.anomaly_feed))
    with kpi2:
        st.metric(label="Valor Monetário das Anomalias", value=f"R$ {st.session_state.anomaly_feed['Amount'].sum():.2f}")
    with kpi3:
        st.metric(label="Taxa de Anomalias", value="0.00%")
    with kpi4:
        st.metric(label="Latência Média da API", value="N/A")

    st.markdown("---")
    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("Feed de Anomalias ao Vivo")
        if not st.session_state.anomaly_feed.empty:
            st.dataframe(st.session_state.anomaly_feed.head(500), use_container_width=True)
        else:
            st.info("Sem anomalias na sessão até o momento.")

    with col2:
        st.subheader("Mapa de Atividade Suspeita")
        if not st.session_state.anomaly_feed.empty:
            st.map(st.session_state.anomaly_feed[["Latitude", "Longitude"]].tail(MAX_MAP_POINTS))
        else:
            st.info("Aguardando detecção de anomalias para exibir no mapa.")

    st.header("Deep Dive da Última Predição")
    if 'last_prediction_result' in st.session_state:
        result = st.session_state.last_prediction_result
        if result.get("success"):
            c1, c2 = st.columns(2)
            with c1:
                st.success(f"**Resultado:** {result.get('prediction_label', 'N/A')}")
                if st.button("Explicar Predição (SHAP)"):
                    with st.spinner("Iniciando análise de explicação..."):
                        explanation_job = explain_transaction(st.session_state.last_prediction_input)
                        if explanation_job.get("success"):
                            st.session_state.explanation_job_id = explanation_job.get('explanation', {}).get('job_id')
                            st.session_state.last_explanation = None
                            st.info("Análise de explicação iniciada. Verifique o resultado em breve.")
                        else:
                            st.error(f"Falha ao iniciar a explicação: {explanation_job.get('error')}")

            with c2:
                st.write("**Dados da Transação Enviada:**")
                st.json(st.session_state.last_prediction_input, expanded=False)

            # Botão manual para buscar o resultado (evita polling agressivo)
            if st.session_state.get('explanation_job_id') and not st.session_state.get('last_explanation'):
                if st.button("Verificar Resultado da Explicação"):
                    with st.spinner("Buscando resultado..."):
                        r = get_explanation_result(st.session_state.explanation_job_id)
                        if r.get("success"):
                            st.session_state.last_explanation = {"success": True, "explanation": r.get("explanation")}
                            st.session_state.explanation_job_id = None
                            st.rerun()
                        elif r.get("status_code") == 202:
                            st.info("A análise ainda está em andamento. Tente novamente em alguns segundos.")
                        else:
                            st.error(f"Erro ao buscar resultado: {r.get('error')}")

            if st.session_state.get('last_explanation'):
                exp = st.session_state.last_explanation
                if exp.get("success"):
                    st.plotly_chart(create_waterfall_plot(exp['explanation']), use_container_width=True)
                else:
                    st.error(f"Falha ao gerar explicação: {exp.get('error')}")
        else:
            st.error(result.get("error", "Falha desconhecida na predição."))
    else:
        st.info("Clique em 'Testar Predição com Amostra' para ver os detalhes.")

# -----------------------------------------------------------------------------
# Tab 2 — Investigativa (Histórico)
# -----------------------------------------------------------------------------
with tab2:
    st.header("Análise Tática de Dados Históricos")

    st.subheader("Explorador de Dados (paginado)")
    filtered_data, total_records = fetch_filtered_data_streaming(
        PARQUET_PATH, amount_range, income_range, st.session_state.page_num, PAGE_SIZE
    )

    if not filtered_data.empty:
        st.dataframe(filtered_data, use_container_width=True)
        st.info(f"Mostrando {len(filtered_data)} de {total_records} registros filtrados.")

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
        sample = filtered_data.sample(min(PLOT_SAMPLE, len(filtered_data)), random_state=42)
        c1, c2 = st.columns(2)
        with c1:
            fig_hist = px.histogram(sample, x="transaction_hour", title="Distribuição por Hora (Amostra)")
            st.plotly_chart(fig_hist, use_container_width=True)
        with c2:
            fig_box = px.box(sample, y="amount", title="Distribuição de Valores (Amostra)")
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

    report_dir = "outputs/validation/drift_detection"
    if os.path.exists(report_dir):
        report_files = [f for f in os.listdir(report_dir) if f.endswith(".html")]
        if report_files:
            selected_report = st.selectbox("Selecione um relatório para visualizar", report_files)
            if selected_report:
                with open(os.path.join(report_dir, selected_report), 'r', encoding='utf-8') as f:
                    html_content = f.read()
                components.html(html_content, height=600, scrolling=True)
        else:
            st.info("Nenhum relatório de drift encontrado.")
    else:
        st.warning(f"Diretório de relatórios não encontrado: {report_dir}")