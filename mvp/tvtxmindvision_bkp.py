import streamlit as st
import cv2
import tempfile
import mediapipe as mp
import time
import torch
import pickle
import numpy as np
import os

# Verifica se CUDA está disponível
dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
st.sidebar.write(f"Usando dispositivo: {dispositivo}")

# Inicializa o MediaPipe Pose e Drawing Utils
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Função para detectar o tipo de golpe com base nos marcos do corpo
def detectar_golpe(marcos):
    if marcos:
        ombro_direito = marcos[12]
        cotovelo_direito = marcos[14]
        pulso_direito = marcos[16]
        
        # Cálculo do ângulo entre o ombro, cotovelo e pulso
        ombro_para_cotovelo = np.array([cotovelo_direito[0] - ombro_direito[0], cotovelo_direito[1] - ombro_direito[1]])
        cotovelo_para_pulso = np.array([pulso_direito[0] - cotovelo_direito[0], pulso_direito[1] - cotovelo_direito[1]])
        
        angulo = np.degrees(np.arccos(
            np.dot(ombro_para_cotovelo, cotovelo_para_pulso) / (np.linalg.norm(ombro_para_cotovelo) * np.linalg.norm(cotovelo_para_pulso))
        ))

        if angulo < 45 and pulso_direito[1] < ombro_direito[1]:
            return "Saque"
        elif angulo > 100 and pulso_direito[0] > ombro_direito[0]:
            return "Forehand"
        elif angulo > 100 and pulso_direito[0] < ombro_direito[0]:
            return "Backhand"

    return "Desconhecido"

# Função para processar o vídeo, detectar golpes e salvar os marcos em um arquivo pickle
def processar_video(cap, gravar=False):
    fps = 30  # Define a taxa de quadros como 30 FPS para reprodução em tempo real
    stframe = st.empty()  # Placeholder para os quadros do vídeo
    dados_marcos = {}

    # Preparação para gravar o vídeo, se necessário
    gravador_video = None
    if gravar:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        gravador_video = cv2.VideoWriter("video_gravado.avi", fourcc, fps, (640, 480))

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
        numero_quadro = 0
        tempo_inicial = time.time()
        
        while cap.isOpened():
            ret, quadro = cap.read()
            if not ret or (gravar and time.time() - tempo_inicial > 20):
                break

            quadro_rgb = cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB)
            resultados = pose.process(quadro_rgb)

            if resultados.pose_landmarks:
                marcos = {
                    id_marca: (marca.x, marca.y, marca.z, marca.visibility)
                    for id_marca, marca in enumerate(resultados.pose_landmarks.landmark)
                }
                dados_marcos[numero_quadro] = marcos

                tipo_golpe = detectar_golpe(marcos)
                
                mp_drawing.draw_landmarks(
                    quadro,
                    resultados.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                )
                cv2.putText(quadro, f"Golpe: {tipo_golpe}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            stframe.image(quadro, channels='BGR', use_container_width=True)
            if gravar:
                gravador_video.write(quadro)
            numero_quadro += 1
            #time.sleep(1.0 / fps)

    cap.release()
    if gravador_video:
        gravador_video.release()
    
    with open("dados_marcos_pose.pkl", "wb") as f:
        pickle.dump(dados_marcos, f)
    st.sidebar.success("Os dados dos marcos foram salvos.")

    if gravar:
        with open("video_gravado.avi", "rb") as f:
            st.sidebar.download_button(
                label="Baixar vídeo gravado de 20 segundos",
                data=f,
                file_name="video_gravado.avi",
                mime="video/x-msvideo"
            )

    with open("dados_marcos_pose.pkl", "rb") as f:
        st.sidebar.download_button(
            label="Baixar Dados dos Marcos de Pose",
            data=f,
            file_name="dados_marcos_pose.pkl",
            mime="application/octet-stream"
        )

    os.remove("dados_marcos_pose.pkl")

# Interface do aplicativo Streamlit
st.title("TVTxMindVision - Detecção de Pose e Reconhecimento de Golpes")

# Instruções para o usuário
st.markdown("""
### Instruções:
1. **Selecione a Fonte de Vídeo**: Use a barra lateral para escolher entre fazer upload de um vídeo ou usar a câmera ao vivo.
2. **Para Vídeos Enviados**: Simplesmente faça upload do arquivo de vídeo e ele será processado automaticamente.
3. **Para Gravação com Webcam**: Clique em "Iniciar Gravação com Webcam ao Vivo" para gravar um vídeo de 20 segundos com sua webcam.
4. **Detecção de Golpes**: O aplicativo irá detectar e exibir o tipo de golpe (Saque, Forehand, Backhand) em tempo real.
5. **Opções de Download**: Após o processamento, baixe os dados dos marcos e o vídeo gravado (se aplicável) da barra lateral.
""")

# Barra lateral para seleção da fonte
fonte = st.sidebar.selectbox("Selecione a Fonte de Vídeo", ("Fazer upload de um vídeo", "Webcam ao Vivo"))

if fonte == "Fazer upload de um vídeo":
    arquivo_enviado = st.sidebar.file_uploader("Envie um arquivo de vídeo", type=["mp4", "mov", "avi"])
    if arquivo_enviado is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(arquivo_enviado.read())
        tfile.close()
        
        cap = cv2.VideoCapture(tfile.name)
        st.text("Processando vídeo enviado, por favor aguarde...")
        processar_video(cap)
        
else:
    if st.sidebar.button("Iniciar Gravação com Webcam ao Vivo (20 segundos)"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Não foi possível abrir a webcam.")
        else:
            st.text("Processando feed da webcam ao vivo, por favor aguarde...")
            processar_video(cap, gravar=True)