import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import mediapipe as mp
import numpy as np
from collections import Counter

# Setup for MediaPipe
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

# Function to process the video frames
def process_frame(frame):
    global stroke_counter

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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

    return frame

# Function to handle the video stream from webrtc_streamer
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    frame = frame.to_ndarray(format="bgr24")
    frame = process_frame(frame)
    return av.VideoFrame.from_ndarray(frame, format="bgr24")

# CSS for responsiveness and styling
st.markdown(
    """
    <style>
        .app-title {
            font-size: 24px;
            font-weight: bold;
            color: #1E3A8A;
            text-align: center;
            margin-bottom: 20px;
            font-family: 'Arial', sans-serif;
        }

        .sidebar .sidebar-content {
            padding-top: 0px;
        }

        .sidebar .stDataFrame, 
        .sidebar .stMarkdown,
        .sidebar .stImage {
            margin-bottom: 10px;
        }

        /* Styling for the stroke statistics table */
        .stDataFrame table {
            border-collapse: collapse;
            width: 100%;
        }

        .stDataFrame th, .stDataFrame td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }

        .stDataFrame th {
            background-color: #f2f2f2;
        }

        /* Styling for the bar chart */
        .stBarChart {
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add decoration with the App Name
st.markdown(
    """
    <div class="app-title">
        BMDS®MindVision
    </div>
""",
    unsafe_allow_html=True,
)

# Instructions for the user
st.markdown(
    """
### Instruções:
1. **Permita o acesso à câmera** quando solicitado pelo navegador.
2. **Posicione-se** em frente à câmera de forma que seu corpo fique visível.
3. **Execute os golpes de badminton** que deseja analisar.
4. **Observe a detecção dos golpes** em tempo real na tela principal.
5. **Acompanhe as estatísticas** dos golpes na barra lateral.
"""
)

# WebRTC Streamer
webrtc_streamer(
    key="stroke_detection",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

# Sidebar for displaying statistics and chart
with st.sidebar:
    st.title("🏸 Análise de Golpes")

    # Calculate total strokes
    total_count = sum(stroke_counter.values())

    if total_count > 0:
        # Display detailed information with percentages in a table
        stats_df = {
            'Golpe': list(stroke_counter.keys()),
            'Contagem': list(stroke_counter.values()),
            'Porcentagem': [f"{(count / total_count) * 100:.1f}%" for count in stroke_counter.values()]
        }
        st.write("### Detalhamento dos Golpes")
        st.dataframe(stats_df, use_container_width=True)

        # Display the bar chart
        st.write("### Representação Gráfica")
        st.bar_chart([count / total_count for count in stroke_counter.values()])
    else:
        st.info("Nenhum golpe detectado ainda.")