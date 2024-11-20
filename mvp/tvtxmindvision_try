import streamlit as st
import cv2
import tempfile
import mediapipe as mp
import numpy as np
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
def process_video(cap):
    global stroke_counter
    stroke_counter.clear()

    stframe = st.image([])  # Placeholder for video frames
    stats_frame = st.sidebar.empty()  # Sidebar for stroke statistics

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
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

            # Update frames in Streamlit
            stframe.image(frame, channels="BGR")

            # Update stroke counts in the sidebar
            stats_frame.write(f"### Stroke Statistics\n{dict(stroke_counter)}")

    cap.release()

# Streamlit app interface
st.title("Real-Time Stroke Detection and Statistics")
st.sidebar.title("Controls")

# Select video source
video_source = st.sidebar.selectbox("Choose video source:", ["Upload a video", "Webcam live feed"])

if video_source == "Upload a video":
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    if uploaded_file:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        temp_file.close()
        
        cap = cv2.VideoCapture(temp_file.name)
        st.text("Processing uploaded video...")
        process_video(cap)
else:
    if st.sidebar.button("Start Webcam"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access the webcam.")
        else:
            st.text("Processing live webcam feed...")
            process_video(cap)
