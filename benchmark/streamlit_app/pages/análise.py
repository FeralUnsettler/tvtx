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
    st.title("Benchmark de Análise de Vídeo")
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
