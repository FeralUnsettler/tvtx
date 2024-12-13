import streamlit as st
import cv2
import tempfile
import mediapipe as mp
import numpy as np
import pandas as pd
import time
from collections import Counter
from camera_input_live import camera_input_live
import matplotlib.pyplot as plt

# Configurações do MediaPipe e inicialização dos contadores
mp_pose = mp.solutions.pose  # Solução de pose do MediaPipe
mp_drawing = mp.solutions.drawing_utils  # Utilitário para desenhar landmarks
stroke_counter = Counter()  # Contador para golpes reconhecidos

# Função para detectar golpes com base nos landmarks
# Analisa o ângulo entre o ombro, cotovelo e pulso para determinar o tipo de golpe
def detect_stroke(landmarks):
    if landmarks:
        right_shoulder = landmarks[12]
        right_elbow = landmarks[14]
        right_wrist = landmarks[16]

        # Vetores para calcular o ângulo
        shoulder_to_elbow = np.array([right_elbow.x - right_shoulder.x, right_elbow.y - right_shoulder.y])
        elbow_to_wrist = np.array([right_wrist.x - right_elbow.x, right_wrist.y - right_elbow.y])

        # Cálculo do ângulo entre os vetores
        angle = np.degrees(np.arccos(
            np.dot(shoulder_to_elbow, elbow_to_wrist) /
            (np.linalg.norm(shoulder_to_elbow) * np.linalg.norm(elbow_to_wrist) + 1e-6)
        ))

        # Condições para identificar o tipo de golpe
        if angle < 45 and right_wrist.y < right_shoulder.y:
            return "Saque"
        elif angle > 100 and right_wrist.x > right_shoulder.x:
            return "Forehand"
        elif angle > 100 and right_wrist.x < right_shoulder.x:
            return "Backhand"

    return "Desconhecido"

# Função para processar imagens ao vivo da câmera
# Analisa cada frame, detecta golpes e QR codes, e exibe os resultados
def process_live_camera():
    global stroke_counter
    stroke_counter.clear()  # Reinicia o contador

    stframe = st.image([])  # Espaço reservado para o feed no Streamlit
    stats_frame = st.sidebar.empty()  # Espaço reservado para estatísticas na barra lateral

    video_writer = None  # Para gravação de vídeo
    detector = cv2.QRCodeDetector()  # Inicializa o detector de QR code

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            # Captura imagem ao vivo da câmera
            image = camera_input_live()
            if image is None:
                continue

            # Converte bytes da imagem para o formato utilizável pelo OpenCV
            bytes_data = image.getvalue()
            frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            # Inicializa o gravador de vídeo na primeira iteração
            if video_writer is None:
                height, width, _ = frame.shape
                video_writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), 10, (width, height))

            # Converte o quadro para RGB e processa os landmarks
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                stroke_type = detect_stroke(landmarks)  # Detecta o tipo de golpe
                stroke_counter[stroke_type] += 1  # Atualiza o contador

                # Desenha landmarks e o tipo de golpe no quadro
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                )
                cv2.putText(frame, f"Golpe: {stroke_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Detecta QR codes no quadro
            data, bbox, _ = detector.detectAndDecode(frame)
            if data:
                cv2.putText(frame, f"QR Code Detectado: {data}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Escreve o frame processado no vídeo
            video_writer.write(frame)

            stframe.image(frame, channels="BGR")  # Atualiza o feed no Streamlit

            # Atualiza estatísticas na barra lateral com DataFrame e gráfico
            stats_df = pd.DataFrame.from_dict(stroke_counter, orient='index', columns=['Count'])
            stats_df['Percentage'] = (stats_df['Count'] / stats_df['Count'].sum()) * 100
            stats_frame.write("### Estatísticas de Golpes")
            stats_frame.write(stats_df)

            # Gráfico de barras interativo
            st.sidebar.bar_chart(stats_df['Count'])

        video_writer.release()  # Libera o gravador após o loop

# Função para exibir vídeo processado
# Adiciona funcionalidade de download do vídeo e estatísticas
def display_processed_video(video_file):
    st.video(video_file)

    with open(video_file, "rb") as f:
        st.sidebar.download_button("Baixar Vídeo Processado", f, file_name="video_processado.avi")

    stats_df = pd.DataFrame.from_dict(stroke_counter, orient='index', columns=['Count'])
    stats_df['Percentage'] = (stats_df['Count'] / stats_df['Count'].sum()) * 100
    csv = stats_df.to_csv().encode('utf-8')
    st.sidebar.download_button("Baixar Estatísticas de Golpes", csv, file_name="estatisticas_golpes.csv")

# Configuração da interface do Streamlit
# Adiciona o logo e instruções
logo_image = "mvp/img/logo.png"  # Substitua pelo caminho do seu logo
st.image(logo_image, use_column_width=True)

st.markdown("""
### Instruções:
1. Escolha entre **enviar um vídeo** ou usar a **webcam ao vivo**.
2. Para vídeos, faça o upload e aguarde o processamento.
3. Para a webcam, clique em "Iniciar Processamento ao Vivo".
4. Veja os golpes detectados em tempo real.
5. Faça o download dos resultados após o processamento.
""")

# Seleção da fonte de vídeo
video_source = st.sidebar.selectbox("Escolha a Fonte de Vídeo", ["Upload de Vídeo", "Webcam ao Vivo"])

if video_source == "Upload de Vídeo":
    uploaded_file = st.sidebar.file_uploader("Envie um arquivo de vídeo", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        st.text("Processando vídeo...")
        video_file = "processed_video.avi"  # Arquivo de saída para vídeo processado

        # Processa vídeo frame a frame
        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            video_writer = None
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    )

                if video_writer is None:
                    height, width, _ = frame.shape
                    video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*"XVID"), 10, (width, height))

                video_writer.write(frame)

            video_writer.release()
        cap.release()

        display_processed_video(video_file)
else:
    if st.sidebar.button("Iniciar Processamento ao Vivo"):
        st.text("Processando feed ao vivo...")
        process_live_camera()
