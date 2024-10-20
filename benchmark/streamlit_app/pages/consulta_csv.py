import streamlit as st
import pandas as pd

def run_consulta_csv():
    st.title("Consultar Resultados de um Arquivo CSV")
    st.markdown("### Carregue um arquivo CSV para visualizar os resultados")

    csv_file = st.file_uploader("Carregue um arquivo CSV", type=["csv"])

    if csv_file is not None:
        results_df = pd.read_csv(csv_file)
        st.write("Resultados carregados:")
        st.write(results_df)
