import streamlit as st
import requests
import pandas as pd
import time
import os
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

# =====================================================================================
# Configurações da Página e Título
# =====================================================================================
st.set_page_config(
    page_title="TrustShield Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ TrustShield - Dashboard de Monitoramento de Fraudes")
st.markdown("Bem-vindo ao painel de controle para detecção de anomalias financeiras.")

# =====================================================================================
# Carregamento de Dados e Estado da Aplicação
# =====================================================================================

@st.cache_data
def load_data(file_path):
    """Carrega os dados do arquivo parquet."""
    try:
        return pd.read_parquet(file_path)
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {file_path}. Certifique-se de que o caminho está correto.")
        return None

# Carrega o dataset histórico para análise
historical_data = load_data("data/features/featured_dataset.parquet")

if 'anomaly_feed' not in st.session_state:
    st.session_state.anomaly_feed = pd.DataFrame(columns=[
        "Timestamp", "Label", "Score", "Amount", "Latitude", "Longitude"
    ])

# =====================================================================================
# Funções de Interação com a API
# =====================================================================================
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

def get_api_status():
    """Verifica o status da API."""
    try:
        response = requests.get(f"{API_URL}/status")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "UNAVAILABLE", "error": str(e)}

def predict_transaction(transaction_data):
    """Envia uma transação para a API e retorna a predição."""
    try:
        response = requests.post(f"{API_URL}/predict", json=transaction_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}

def explain_transaction(transaction_data):
    """Envia uma transação para a API e retorna a explicação SHAP."""
    try:
        response = requests.post(f"{API_URL}/explain", json=transaction_data)
        response.raise_for_status()
        return {"success": True, "explanation": response.json()}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}

def validate_model():
    """Aciona a validação do modelo na API."""
    try:
        response = requests.post(f"{API_URL}/validate")
        response.raise_for_status()
        return {"success": True, "data": response.json()}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}

# =====================================================================================
# Funções de Visualização
# =====================================================================================
def create_waterfall_plot(explanation):
    """Cria um gráfico de cascata (waterfall) para os valores SHAP."""
    shap_values = explanation['shap_values'][0]
    base_value = explanation['base_values'][0]
    feature_names = explanation['feature_names']
    
    fig = go.Figure(go.Waterfall(
        name = "20", orientation = "v",
        measure = ["relative"] * len(feature_names),
        x = feature_names,
        textposition = "outside",
        text = [f"{val:.2f}" for val in shap_values],
        y = shap_values,
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        base = base_value
    ))

    fig.update_layout(
            title = "Análise de Contribuição das Features (SHAP)",
            showlegend = True
    )
    return fig

# =====================================================================================
# Layout da Barra Lateral (Sidebar)
# =====================================================================================
with st.sidebar:
    st.header("Status da API")
    if st.button("Atualizar Status"):
        st.session_state.api_status = get_api_status()
    
    if 'api_status' not in st.session_state:
        st.session_state.api_status = get_api_status()

    api_status = st.session_state.api_status
    if api_status.get("status") == "OPERATIONAL":
        st.success(f"**Status:** {api_status.get('status')}")
        st.write(f"**Modelo:** {api_status.get('model_type')}")
    else:
        st.error(f"**Status:** {api_status.get('status', 'ERRO')}")
        st.caption(api_status.get('error', 'API indisponível.'))

    st.header("Ações")
    if st.button("Testar Predição com Amostra"):
        sample_data = generate_sample_transaction()
        st.session_state.last_prediction_input = sample_data
        prediction = predict_transaction(sample_data)
        st.session_state.last_prediction_result = prediction
        st.session_state.last_explanation = None # Limpa a explicação anterior
        
        if prediction.get("success") and prediction.get("prediction_label") == "ANOMALIA":
            new_entry = pd.DataFrame([{
                "Timestamp": pd.to_datetime(prediction.get("timestamp")),
                "Label": prediction.get("prediction_label"),
                "Score": prediction.get("confidence_score"),
                "Amount": sample_data.get("amount"),
                "Latitude": sample_data.get("latitude"),
                "Longitude": sample_data.get("longitude"),
            }])
            st.session_state.anomaly_feed = pd.concat([new_entry, st.session_state.anomaly_feed], ignore_index=True)

    if historical_data is not None:
        st.header("Filtros de Análise Histórica")
        amount_range = st.slider(
            "Valor da Transação (Amount)", 
            min_value=float(historical_data['amount'].min()),
            max_value=float(historical_data['amount'].max()),
            value=(float(historical_data['amount'].min()), float(historical_data['amount'].max()))
        )
        
        income_range = st.slider(
            "Renda Anual (Yearly Income)",
            min_value=float(historical_data['yearly_income'].min()),
            max_value=float(historical_data['yearly_income'].max()),
            value=(float(historical_data['yearly_income'].min()), float(historical_data['yearly_income'].max()))
        )

# =====================================================================================
# Abas Principais
# =====================================================================================
tab1, tab2, tab3 = st.tabs(["Monitoramento em Tempo Real", "Análise Investigativa", "Performance do Modelo"])

with tab1:
    st.header("Visão Operacional")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(label="Total de Anomalias (Sessão)", value=len(st.session_state.anomaly_feed))
    with kpi2:
        st.metric(label="Valor Monetário das Anomalias", value=f"R$ {st.session_state.anomaly_feed['Amount'].sum():.2f}")
    with kpi3:
        st.metric(label="Taxa de Anomalias", value="0.00%") # Placeholder
    with kpi4:
        st.metric(label="Latência Média da API", value="N/A") # Placeholder

    st.markdown("---")
    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("Feed de Anomalias ao Vivo")
        feed_placeholder = st.empty()
        with feed_placeholder.container():
            st.dataframe(st.session_state.anomaly_feed, use_container_width=True)
    with col2:
        st.subheader("Mapa de Atividade Suspeita")
        if not st.session_state.anomaly_feed.empty:
            st.map(st.session_state.anomaly_feed[["Latitude", "Longitude"]])
        else:
            st.info("Aguardando detecção de anomalias para exibir no mapa.")

    st.header("Deep Dive da Última Predição")
    if 'last_prediction_result' in st.session_state:
        result = st.session_state.last_prediction_result
        if result.get("success"):
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Resultado:** {result.get('prediction_label', 'N/A')}")
                if st.button("Explicar Predição (SHAP)"):
                    with st.spinner("Gerando explicação..."):
                        explanation_result = explain_transaction(st.session_state.last_prediction_input)
                        st.session_state.last_explanation = explanation_result
            
            with col2:
                st.write("**Dados da Transação Enviada:**")
                st.json(st.session_state.last_prediction_input, expanded=False)

            if st.session_state.get('last_explanation'):
                exp = st.session_state.last_explanation
                if exp.get("success"):
                    st.plotly_chart(create_waterfall_plot(exp['explanation']), use_container_width=True)
                else:
                    st.error(f"Falha ao gerar explicação: {exp.get('error')}")
    else:
        st.info("Clique em 'Testar Predição' para ver os detalhes.")

with tab2:
    st.header("Análise Tática de Dados Históricos")
    if historical_data is not None:
        st.subheader("Explorador de Dados")
        
        # Aplicar filtros
        filtered_data = historical_data[
            (historical_data['amount'] >= amount_range[0]) & (historical_data['amount'] <= amount_range[1]) &
            (historical_data['yearly_income'] >= income_range[0]) & (historical_data['yearly_income'] <= income_range[1])
        ]
        
        st.dataframe(filtered_data.head(1000), use_container_width=True)
        st.info(f"Mostrando {min(1000, len(filtered_data))} de {len(filtered_data)} registros filtrados.")
        
        st.markdown("---")
        st.subheader("Visualizações Interativas")
        
        # Gráficos com Plotly
        c1, c2 = st.columns(2)
        with c1:
            fig_hist = px.histogram(filtered_data, x="transaction_hour", 
                                    title="Distribuição de Transações por Hora",
                                    color_discrete_sequence=px.colors.qualitative.Prism)
            st.plotly_chart(fig_hist, use_container_width=True)

        with c2:
            fig_box = px.box(filtered_data, y="amount", 
                               title="Distribuição de Valores de Transação",
                               color_discrete_sequence=px.colors.qualitative.Safe)
            st.plotly_chart(fig_box, use_container_width=True)

    else:
        st.warning("Não foi possível carregar os dados históricos para análise.")

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
