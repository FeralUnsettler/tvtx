import streamlit as st
import cv2
import tempfile
import mediapipe as mp
import numpy as np
import pickle
import os
import time
from collections import Counter

# Setup for MediaPipe and video processing
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize counters for recognized strokes
stroke_counter = Counter()

# Define function for stroke detection
def detect_stroke(landmarks):
    if landmarks:
        right_shoulder = landmarks[12]
        right_elbow = landmarks[14]
        right_wrist = landmarks[16]

        shoulder_to_elbow = np.array([right_elbow.x - right_shoulder.x, right_elbow.y - right_shoulder.y])
        elbow_to_wrist = np.array([right_wrist.x - right_elbow.x, right_wrist.y - right_elbow.y])
        
        angle = np.degrees(np.arccos(
            np.dot(shoulder_to_elbow, elbow_to_wrist) / 
            (np.linalg.norm(shoulder_to_elbow) * np.linalg.norm(elbow_to_wrist) + 1e-6)
        ))

        if angle < 45 and right_wrist.y < right_shoulder.y:
            return "Serve"
        elif angle > 100 and right_wrist.x > right_shoulder.x:
            return "Forehand"
        elif angle > 100 and right_wrist.x < right_shoulder.x:
            return "Backhand"

    return "Unknown"

# Function to process video and count strokes
def process_video(cap, record=False):
    global stroke_counter
    stroke_counter.clear()

    video_frames = []  # Ensure video_frames is initialized
    stframe = st.image([])  # Placeholder for video frames
    stats_frame = st.sidebar.empty()  # Sidebar for stroke statistics

    start_time = time.time()  # Start the timer to stop after 20 seconds

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            elapsed_time = time.time() - start_time
            if elapsed_time >= 20:  # Stop after 20 seconds
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                stroke_type = detect_stroke(landmarks)
                stroke_counter[stroke_type] += 1

                # Draw landmarks and stroke type on the frame
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                )
                cv2.putText(frame, f"Stroke: {stroke_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Collect frames for later download
            video_frames.append(frame)

            # Update frames in Streamlit
            stframe.image(frame, channels="BGR")

            # Update stroke counts in the sidebar
            stats_frame.write(f"### Stroke Statistics\n{dict(stroke_counter)}")

    cap.release()

    # Save video and pickle if recording
    if record:
        video_file = save_video(video_frames)
        pickle_file = save_pickle(stroke_counter)
        return video_file, pickle_file
    return None, None

# Save video to a temporary file
def save_video(video_frames):
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_video.name, cv2.VideoWriter_fourcc(*'H264'), 20.0, (640, 480))
    for frame in video_frames:
        out.write(frame)
    out.release()
    return temp_video.name

# Save stroke data as pickle
def save_pickle(data):
    temp_pickle = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    with open(temp_pickle.name, 'wb') as f:
        pickle.dump(data, f)
    return temp_pickle.name

# Function to display the processed video in Streamlit
def display_processed_video(video_path):
    video_file = open(video_path, 'rb')  # Open the video file in binary mode
    video_bytes = video_file.read()  # Read the file's bytes
    st.video(video_bytes)  # Display the video in Streamlit

# Add decoration with the App Name
st.markdown("""
    <style>
        .app-title {
            font-size: 32px;
            font-weight: bold;
            color: #1E3A8A;  /* Dark blue */
            text-align: center;
            margin-bottom: 20px;
            font-family: 'Arial', sans-serif;
        }
    </style>
    <div class="app-title">
        BMDS®MindVision
    </div>
""", unsafe_allow_html=True)

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
video_source = st.sidebar.selectbox("Selecione a Fonte de Vídeo", ("Fazer upload de um vídeo", "Webcam ao Vivo"))

if video_source == "Fazer upload de um vídeo":
    uploaded_file = st.sidebar.file_uploader("Envie um arquivo de vídeo", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()
        
        cap = cv2.VideoCapture(tfile.name)
        st.text("Processando vídeo enviado, por favor aguarde...")
        video_file, pickle_file = process_video(cap, record=False)

        # Cálculo do total de golpes
        total_count = sum(stroke_counter.values())

        # Adiciona título à barra lateral
        st.sidebar.title("🏸 **Análise de Golpes**")

        # Exibição detalhada com porcentagens em tabela
        stats_df = {
            'Golpe': list(stroke_counter.keys()),
            'Contagem': list(stroke_counter.values()),
            'Porcentagem': [f"{(count / total_count) * 100:.1f}%" for count in stroke_counter.values()]
        }
        st.sidebar.write("### Detalhamento dos Golpes")
        st.sidebar.dataframe(stats_df, use_container_width=True)  # Tabela estilizada

        # Exibição do gráfico
        st.sidebar.write("### Representação Gráfica")
        st.sidebar.bar_chart([count / total_count for count in stroke_counter.values()])

        # Botões de download
        if video_file:
            with open(video_file, "rb") as f:
                st.sidebar.download_button("Baixar Vídeo", f, file_name="video.mp4")
        if pickle_file:
            with open(pickle_file, "rb") as f:
                st.sidebar.download_button("Baixar Dados (pickle)", f, file_name="dados.pkl")

        # Exibição do vídeo processado
        st.write("### Vídeo Processado")
        display_processed_video(video_file)

else:
    if st.sidebar.button("Iniciar Gravação com Webcam ao Vivo (20 segundos)"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Não foi possível abrir a webcam.")
        else:
            st.text("Processando feed da webcam ao vivo, por favor aguarde...")
            video_file, pickle_file = process_video(cap, record=True)

            # Cálculo do total de golpes
            total_count = sum(stroke_counter.values())

            # Exibição dos dados no lado
            st.sidebar.title("🏸 **Análise de Golpes**")
            stats_df = {
                'Golpe': list(stroke_counter.keys()),
                'Contagem': list(stroke_counter.values()),
                'Porcentagem': [f"{(count / total_count) * 100:.1f}%" for count in stroke_counter.values()]
            }
            st.sidebar.write("### Detalhamento dos Golpes")
            st.sidebar.dataframe(stats_df, use_container_width=True)  # Tabela estilizada

            # Exibição do gráfico
            st.sidebar.write("### Representação Gráfica")
            st.sidebar.bar_chart([count / total_count for count in stroke_counter.values()])

            # Botões de download
            if video_file:
                with open(video_file, "rb") as f:
                    st.sidebar.download_button("Baixar Vídeo", f, file_name="video.mp4")
            if pickle_file:
                with open(pickle_file, "rb") as f:
                    st.sidebar.download_button("Baixar Dados (pickle)", f, file_name="dados.pkl")

            # Exibição do vídeo processado
            st.write("### Vídeo Processado")

            display_processed_video(video_file)
