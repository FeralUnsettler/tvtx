import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import mediapipe as mp
import numpy as np
from collections import Counter
import pickle
import cv2
from datetime import datetime
import os

# Configurar o estilo CSS
st.markdown(
    f"""
    <style>
        body {{
            background-image: url('/home/luxx/Documents/dev/python/tvtx/mvp/img/7.jpg');
            background-size: cover;
            font-family: Arial, sans-serif;
        }}
        .css-18e3th9 {{
            background: rgba(255, 255, 255, 0.8);
            border-radius: 15px;
            padding: 20px;
        }}
        .css-1cpxqw2 {{
            font-weight: bold;
            font-size: 24px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
        }}
        th, td {{
            text-align: left;
            padding: 8px;
            border: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Configuração do WebRTC
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Contador de golpes
stroke_counter = Counter()
recording_start_time = None

# Classe personalizada para processamento de vídeo
class StrokeDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.stroke_counter = Counter()
        self.frames = []

    def recv(self, frame):
        global recording_start_time
        if recording_start_time is None:
            recording_start_time = datetime.now()

        elapsed_time = (datetime.now() - recording_start_time).seconds
        if elapsed_time > 20:
            return frame

        # Convert frame to RGB
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.frames.append(img)

        # Process frame with MediaPipe
        results = self.pose.process(img_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            stroke_type = self.detect_stroke(landmarks)
            self.stroke_counter[stroke_type] += 1

            # Draw landmarks and stroke type
            mp_drawing.draw_landmarks(
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )
            cv2.putText(img, f"Stroke: {stroke_type}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return frame.from_ndarray(img, format="bgr24")

    def detect_stroke(self, landmarks):
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

# Streamlit UI
st.title("🎾 BMDS®MindVision")
st.markdown("**Análise de golpes de tênis em tempo real com visualização aprimorada.**")

# Inicialização do WebRTC
webrtc_ctx = webrtc_streamer(
    key="stroke-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=StrokeDetectionProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

if webrtc_ctx.video_processor:
    processor = webrtc_ctx.video_processor
    total_strokes = sum(processor.stroke_counter.values())

    if total_strokes > 0:
        # Sidebar com estatísticas
        st.sidebar.title("📊 Estatísticas")
        st.sidebar.write("### Contagem de Golpes")
        for stroke, count in processor.stroke_counter.items():
            st.sidebar.write(f"{stroke}: {count} ({(count / total_strokes) * 100:.1f}%)")
        st.sidebar.bar_chart({stroke: count / total_strokes for stroke, count in processor.stroke_counter.items()})

        # Tabela formatada
        st.markdown("### Dados de Golpes")
        st.markdown(
            """
            <table>
                <tr>
                    <th>Tipo de Golpe</th>
                    <th>Quantidade</th>
                    <th>Porcentagem</th>
                </tr>
            """ + "".join(
                f"""
                <tr>
                    <td>{stroke}</td>
                    <td>{count}</td>
                    <td>{(count / total_strokes) * 100:.1f}%</td>
                </tr>
                """ for stroke, count in processor.stroke_counter.items()
            ) + "</table>",
            unsafe_allow_html=True
        )

    # Botões de download
    if st.button("📥 Baixar Vídeo"):
        video_path = "output_video.avi"
        frame_width = processor.frames[0].shape[1]
        frame_height = processor.frames[0].shape[0]
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

        for frame in processor.frames:
            out.write(frame)
        out.release()

        with open(video_path, "rb") as video_file:
            st.download_button("📥 Clique aqui para baixar o vídeo", video_file, file_name="recorded_video.avi")

    if st.button("📥 Baixar Resultados"):
        pickle_path = "stroke_data.pkl"
        with open(pickle_path, "wb") as pickle_file:
            pickle.dump(processor.stroke_counter, pickle_file)

        with open(pickle_path, "rb") as pickle_file:
            st.download_button("📥 Clique aqui para baixar os resultados", pickle_file, file_name="stroke_data.pkl")
