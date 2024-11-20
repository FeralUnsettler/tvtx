import streamlit as st
import cv2
import tempfile
import mediapipe as mp
import time
import numpy as np
from collections import Counter

# Use CUDA if available
device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
st.sidebar.write(f"Using device: {device}")

# Initialize MediaPipe Pose and Drawing Utils
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to detect the type of stroke based on body landmarks
def detect_stroke(landmarks):
    if landmarks:
        right_shoulder = landmarks[12]
        right_elbow = landmarks[14]
        right_wrist = landmarks[16]

        # Compute the angle between shoulder, elbow, and wrist
        shoulder_to_elbow = np.array([right_elbow[0] - right_shoulder[0], right_elbow[1] - right_shoulder[1]])
        elbow_to_wrist = np.array([right_wrist[0] - right_elbow[0], right_wrist[1] - right_elbow[1]])
        
        angle = np.degrees(np.arccos(
            np.dot(shoulder_to_elbow, elbow_to_wrist) /
            (np.linalg.norm(shoulder_to_elbow) * np.linalg.norm(elbow_to_wrist))
        ))

        if angle < 45 and right_wrist[1] < right_shoulder[1]:
            return "Serve"
        elif angle > 100 and right_wrist[0] > right_shoulder[0]:
            return "Forehand"
        elif angle > 100 and right_wrist[0] < right_shoulder[0]:
            return "Backhand"

    return "Unknown"

# Function to process video and display results in real time
def process_video(video_source, record=False):
    fps = 30  # Target 30 FPS for real-time processing
    stframe = st.empty()  # Placeholder for video frames
    stroke_counter = Counter()

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while video_source.isOpened():
            ret, frame = video_source.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = {
                    idx: (lm.x, lm.y, lm.z, lm.visibility)
                    for idx, lm in enumerate(results.pose_landmarks.landmark)
                }
                stroke_type = detect_stroke(landmarks)
                stroke_counter[stroke_type] += 1

                # Draw landmarks and annotate stroke type
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                )
                cv2.putText(frame, f"Stroke: {stroke_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Update video frame in Streamlit
            stframe.image(frame, channels="BGR", use_container_width=True)

    video_source.release()
    return stroke_counter

# Streamlit App Interface
st.title("TVTxMindVision - Real-time Pose Detection and Stroke Recognition")

st.markdown("""
### Instructions:
1. **Choose Video Source**: Select between uploading a video or using the live webcam.
2. **For Uploaded Videos**: Upload the file and it will be processed automatically.
3. **For Webcam Feed**: Start the live webcam feed to analyze strokes in real-time.
4. **Stroke Detection**: The app recognizes strokes like Serve, Forehand, and Backhand.
5. **Statistics**: Download the count of recognized strokes after processing.
""")

# Sidebar for video source selection
video_source = st.sidebar.selectbox("Select Video Source", ("Upload a Video", "Live Webcam"))

if video_source == "Upload a Video":
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()
        
        cap = cv2.VideoCapture(tfile.name)
        st.text("Processing uploaded video, please wait...")
        stats = process_video(cap)
        
        st.sidebar.write("### Stroke Counts")
        for stroke, count in stats.items():
            st.sidebar.write(f"{stroke}: {count}")

else:
    if st.sidebar.button("Start Live Webcam (20 seconds)"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Unable to open webcam.")
        else:
            st.text("Processing live webcam feed, please wait...")
            stats = process_video(cap)
            
            st.sidebar.write("### Stroke Counts")
            for stroke, count in stats.items():
                st.sidebar.write(f"{stroke}: {count}")
