import streamlit as st

# Título do aplicativo
# Configuração da interface do Streamlit
# Adiciona o logo e instruções
logo_image = "mvp/img/logo.png"  # Substitua pelo caminho do seu logo
st.image(logo_image, use_column_width=True)

st.set_page_config(page_title="Benchmark TVTx®MindVision vs. Baseline®Vision", layout="wide")
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
