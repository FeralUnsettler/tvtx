import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import pickle
import time
import os
from collections import Counter

# Setup MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Utility to manage global state for the session
class SessionState:
    def __init__(self):
        self.stroke_counter = Counter()
        self.video_frames = []

# Initialize session state
if "state" not in st.session_state:
    st.session_state.state = SessionState()

# Function to detect stroke based on wrist and foot positions
def detect_stroke(landmarks):
    right_wrist = landmarks[16]
    right_foot = landmarks[32]
    left_foot = landmarks[31]

    if right_wrist.y > max(right_foot.y, left_foot.y):
        return None  # Wrist is below feet, no stroke in progress

    right_shoulder = landmarks[12]
    right_elbow = landmarks[14]

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
    return None

# Process video and track strokes
def process_video(cap, live_feed=False):
    state = st.session_state.state
    state.stroke_counter.clear()
    state.video_frames = []

    start_time = time.time() if live_feed else None
    last_wrist_below_feet = True
    detected_stroke = None

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Stop live feed after 20 seconds
            if live_feed and time.time() - start_time > 20:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                wrist_below_feet = landmarks[16].y > max(landmarks[31].y, landmarks[32].y)

                if not wrist_below_feet and last_wrist_below_feet:
                    # Detect stroke only when wrist moves above feet
                    detected_stroke = detect_stroke(landmarks)
                    if detected_stroke:
                        state.stroke_counter[detected_stroke] += 1

                last_wrist_below_feet = wrist_below_feet

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )
                if detected_stroke:
                    cv2.putText(frame, f"Stroke: {detected_stroke}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            state.video_frames.append(frame)

    cap.release()

# Save video to temporary file
def save_video():
    state = st.session_state.state
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_video.name, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480))
    for frame in state.video_frames:
        out.write(frame)
    out.release()
    return temp_video.name

# Save stroke data as pickle
def save_pickle():
    state = st.session_state.state
    temp_pickle = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    with open(temp_pickle.name, 'wb') as f:
        pickle.dump(state.stroke_counter, f)
    return temp_pickle.name

# Display processed video
def display_processed_video(video_path):
    with open(video_path, "rb") as video_file:
        st.video(video_file.read())

# Main interface
st.markdown("<div style='text-align: center; font-size: 32px; font-weight: bold;'>BMDS®MindVision</div>", unsafe_allow_html=True)

video_source = st.sidebar.selectbox("Fonte de vídeo", ["Upload de vídeo", "Webcam ao vivo"])
if video_source == "Upload de vídeo":
    uploaded_file = st.sidebar.file_uploader("Envie um vídeo", type=["mp4", "mov", "avi"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        st.text("Processando vídeo, aguarde...")
        process_video(cap)

        # Exibir resultados
        video_path = save_video()
        st.write("### Resultados")
        display_processed_video(video_path)
elif st.sidebar.button("Iniciar gravação com webcam (20s)"):
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        st.text("Processando feed da webcam, aguarde...")
        process_video(cap, live_feed=True)
        video_path = save_video()
        st.write("### Resultados")
        display_processed_video(video_path)
    else:
        st.error("Erro ao acessar a webcam.")

# Exibir estatísticas e opções de download
state = st.session_state.state
if state.stroke_counter:
    st.sidebar.write("### Estatísticas de Golpes")
    st.sidebar.bar_chart(state.stroke_counter)
    video_path = save_video()
    with open(video_path, "rb") as f:
        st.sidebar.download_button("Baixar Vídeo", f, "video.mp4")
    pickle_path = save_pickle()
    with open(pickle_path, "rb") as f:
        st.sidebar.download_button("Baixar Dados (pickle)", f, "dados.pkl")
