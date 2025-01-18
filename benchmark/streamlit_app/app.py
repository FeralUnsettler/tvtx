import streamlit as st
from PIL import Image
# Título do aplicativo
# Configuração da interface do Streamlit
# Adiciona o logo e instruções
# Carrega a imagem usando PIL
logo_image = Image.open("benchmark/streamlit_app/logo.png")  # Substitua pelo caminho da sua imagem
st.set_page_config(page_title="Benchmark TVTx®MindVision vs. Baseline®Vision", layout="wide")
st.image(logo_image, use_container=True, caption="TVTx®MindVision", width=200)
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
