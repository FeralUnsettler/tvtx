import streamlit as st

# Título do aplicativo
st.set_page_config(page_title="Benchmark: Tênis Vídeo Treino vs. SwingVision", layout="wide")
st.sidebar.title("Navegação")
st.sidebar.markdown("### Escolha uma página:")
page = st.sidebar.radio("Ir para:", ["Análise", "Consultar CSV", "Consultar Banco de Dados"])

# Importar a página correspondente
if page == "Análise":
    from pages.análise import run_análise
    run_análise()
elif page == "Consultar CSV":
    from pages.consulta_csv import run_consulta_csv
    run_consulta_csv()
#elif page == "Consultar Banco de Dados":
#    from pages.consulta_db import run_consulta_db()
