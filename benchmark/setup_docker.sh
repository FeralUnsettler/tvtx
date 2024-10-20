#!/bin/bash

# Nome do projeto
PROJECT_NAME="streamlit_app"

# Criar diretórios e arquivos principais
mkdir -p $PROJECT_NAME/pages

# Criar arquivo app.py
cat <<EOL > $PROJECT_NAME/app.py
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
elif page == "Consultar Banco de Dados":
    from pages.consulta_db import run_consulta_db()
EOL

# Criar arquivos para as páginas
cat <<EOL > $PROJECT_NAME/pages/análise.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3

# Funções de análise (simulação)
def analyze_tvtx(video_path):
    return {
        "forehand_accuracy": np.random.rand(),
        "backhand_accuracy": np.random.rand(),
        "serve_accuracy": np.random.rand(),
    }

def analyze_swingvision(video_path):
    return {
        "forehand_accuracy": np.random.rand(),
        "backhand_accuracy": np.random.rand(),
        "serve_accuracy": np.random.rand(),
    }

# Função para salvar resultados em um banco de dados SQLite
def save_results_to_db(tvtx_results, swingvision_results):
    conn = sqlite3.connect('resultados.db')
    conn.execute('''
    CREATE TABLE IF NOT EXISTS resultados (
        id INTEGER PRIMARY KEY,
        forehand_accuracy_tvtx REAL,
        backhand_accuracy_tvtx REAL,
        serve_accuracy_tvtx REAL,
        forehand_accuracy_swingvision REAL,
        backhand_accuracy_swingvision REAL,
        serve_accuracy_swingvision REAL
    )
    ''')
    conn.execute('''
    INSERT INTO resultados (forehand_accuracy_tvtx, backhand_accuracy_tvtx, serve_accuracy_tvtx,
                            forehand_accuracy_swingvision, backhand_accuracy_swingvision, serve_accuracy_swingvision)
    VALUES (?, ?, ?, ?, ?, ?)''', (
        tvtx_results["forehand_accuracy"],
        tvtx_results["backhand_accuracy"],
        tvtx_results["serve_accuracy"],
        swingvision_results["forehand_accuracy"],
        swingvision_results["backhand_accuracy"],
        swingvision_results["serve_accuracy"]
    ))
    conn.commit()
    conn.close()

def run_análise():
    st.title("Análise de Vídeo")
    st.markdown("### Carregue um vídeo de treino para análise")

    # Upload de vídeo
    video_file = st.file_uploader("Carregue um vídeo de treino", type=["mp4", "mov"])

    if video_file is not None:
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(video_file.read())

        # Análise com o Tênis Vídeo Treino
        tvtx_results = analyze_tvtx(video_path)
        st.subheader("Análise com Tênis Vídeo Treino")
        st.write(tvtx_results)

        # Análise com o SwingVision
        swingvision_results = analyze_swingvision(video_path)
        st.subheader("Análise com SwingVision")
        st.write(swingvision_results)

        # Comparação dos resultados
        comparison_df = pd.DataFrame({
            "Atributo": ["Forehand Accuracy", "Backhand Accuracy", "Serve Accuracy"],
            "Tênis Vídeo Treino": [
                tvtx_results["forehand_accuracy"],
                tvtx_results["backhand_accuracy"],
                tvtx_results["serve_accuracy"]
            ],
            "SwingVision": [
                swingvision_results["forehand_accuracy"],
                swingvision_results["backhand_accuracy"],
                swingvision_results["serve_accuracy"]
            ]
        })

        st.write(comparison_df)

        # Salvar em CSV
        csv = comparison_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Baixar resultados como CSV",
            data=csv,
            file_name='resultado_analise.csv',
            mime='text/csv',
        )

        # Salvar em banco de dados
        save_results_to_db(tvtx_results, swingvision_results)

    # Rodapé
    st.markdown("Desenvolvido por Luciano Martins Fagundes")
EOL

cat <<EOL > $PROJECT_NAME/pages/consulta_csv.py
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
EOL

cat <<EOL > $PROJECT_NAME/pages/consulta_db.py
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
EOL

# Criar requirements.txt
cat <<EOL > $PROJECT_NAME/requirements.txt
streamlit
pandas
numpy
matplotlib
EOL

# Criar Dockerfile
cat <<EOL > $PROJECT_NAME/Dockerfile
# Usar uma imagem base do Python
FROM python:3.10-slim

# Definir o diretório de trabalho
WORKDIR /app

# Copiar os arquivos de requisitos e instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o restante dos arquivos do projeto
COPY . .

# Expor a porta do Streamlit
EXPOSE 8501

# Comando para iniciar o aplicativo Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
EOL

# Criar docker-compose.yml
cat <<EOL > $PROJECT_NAME/docker-compose.yml
version: '3.8'

services:
  streamlit:
    build: .
    ports:
      - "8501:8501"
EOL

# Mensagem de conclusão
echo "Estrutura do projeto Dockerizada criada com sucesso!"
echo "Para iniciar o aplicativo, navegue até o diretório '$PROJECT_NAME' e execute 'docker-compose up'."