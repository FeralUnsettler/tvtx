import streamlit as st
import cv2
import tempfile
import mediapipe as mp
import numpy as np
import time
import pickle
from collections import Counter

# Setup para MediaPipe e processamento de vídeo
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Contador para os golpes reconhecidos
stroke_counter = Counter()

# CSS para estilizar o aplicativo
st.markdown("""
    <style>
        body {
            background-image: url('https://via.placeholder.com/1500x1000.png?text=Quadra+de+T%C3%AAnis');
            background-size: cover;
        }
        .main {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 15px;
            padding: 20px;
        }
        .stSidebar {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        table, th, td {
            border: 1px solid #ddd;
        }
        th, td {
            text-align: center;
            padding: 8px;
        }
        th {
            background-color: #007BFF;
            color: white;
        }
        td {
            background-color: #f9f9f9;
        }
    </style>
""", unsafe_allow_html=True)

# Função para detectar golpes
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

# Função para renderizar as estatísticas em tabela
def render_statistics(counter):
    total_count = sum(counter.values())
    stats_table = """
    <table>
        <tr>
            <th>Golpe</th>
            <th>Contagem</th>
            <th>Porcentagem</th>
        </tr>
    """
    for stroke, count in counter.items():
        percentage = (count / total_count) * 100
        stats_table += f"""
        <tr>
            <td>{stroke}</td>
            <td>{count}</td>
            <td>{percentage:.1f}%</td>
        </tr>
        """
    stats_table += "</table>"
    return stats_table

# Função para processar vídeo
def process_video(cap, duration=20):
    global stroke_counter
    stroke_counter.clear()

    stframe = st.image([])  # Placeholder para os frames do vídeo
    stats_frame = st.sidebar.empty()  # Estatísticas na barra lateral

    frames = []
    start_time = time.time()

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                stroke_type = detect_stroke(landmarks)
                stroke_counter[stroke_type] += 1

                # Desenhar landmarks e tipo de golpe no frame
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                )
                cv2.putText(frame, f"Stroke: {stroke_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Atualizar os frames no Streamlit
            stframe.image(frame, channels="BGR")
            frames.append(frame)

            # Atualizar contagem de golpes na barra lateral
            stats_frame.markdown(render_statistics(stroke_counter), unsafe_allow_html=True)

    cap.release()
    return frames

# Instruções para o usuário
st.markdown("""
    ## 🎾 Aplicativo de Análise de Golpes no Tênis
    1. Faça upload de um vídeo ou use a webcam ao vivo.
    2. Observe os golpes detectados em tempo real.
    3. Veja as estatísticas detalhadas na barra lateral.
    4. Após 20 segundos de gravação, baixe o vídeo processado e os dados gerados.
""")

# Barra lateral para seleção da fonte de vídeo
video_source = st.sidebar.selectbox("Selecione a Fonte de Vídeo", ("Fazer upload de um vídeo", "Webcam ao Vivo"))

if video_source == "Fazer upload de um vídeo":
    uploaded_file = st.sidebar.file_uploader("Envie um arquivo de vídeo", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        st.text("Processando vídeo enviado, por favor aguarde...")
        frames = process_video(cap)

        # Salvar vídeo processado
        video_output = "processed_video.mp4"
        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()

        # Salvar dados de contagem
        pickle_output = "stroke_data.pkl"
        with open(pickle_output, 'wb') as f:
            pickle.dump(stroke_counter, f)

        # Botões de download
        st.sidebar.download_button("Baixar Vídeo Processado", open(video_output, "rb"), file_name="processed_video.mp4")
        st.sidebar.download_button("Baixar Dados de Golpes", open(pickle_output, "rb"), file_name="stroke_data.pkl")

elif video_source == "Webcam ao Vivo":
    if st.sidebar.button("Iniciar Gravação com Webcam ao Vivo"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Não foi possível abrir a webcam.")
        else:
            st.text("Gravando vídeo ao vivo por 20 segundos...")
            frames = process_video(cap)

            # Salvar vídeo e dados
            video_output = "processed_video.mp4"
            height, width, _ = frames[0].shape
            out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
            for frame in frames:
                out.write(frame)
            out.release()

            pickle_output = "stroke_data.pkl"
            with open(pickle_output, 'wb') as f:
                pickle.dump(stroke_counter, f)

            # Botões de download
            st.sidebar.download_button("Baixar Vídeo Processado", open(video_output, "rb"), file_name="processed_video.mp4")
            st.sidebar.download_button("Baixar Dados de Golpes", open(pickle_output, "rb"), file_name="stroke_data.pkl")
