import streamlit as st
import pandas as pd
import sqlite3

def query_results_from_db():
    conn = sqlite3.connect('resultados.db')
    query = "SELECT * FROM resultados"
    results_df = pd.read_sql(query, conn)
    conn.close()
    return results_df

def run_consulta_db():
    st.title("Consultar Resultados do Banco de Dados")
    st.markdown("### Visualize os resultados salvos no banco de dados")

    if st.button("Carregar Resultados"):
        results_df = query_results_from_db()
        st.write("Resultados carregados:")
        st.write(results_df)
