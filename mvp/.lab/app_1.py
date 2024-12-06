import streamlit as st
import cv2
import tempfile
import mediapipe as mp
import time
import torch  # To ensure GPU availability

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.write(f"Using device: {device}")

# Initialize MediaPipe pose landmarker
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to process video and draw landmarks, presenting it as a continuous video stream
def process_video(file):
    # Create a temporary file to save the uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(tfile.name)
    
    # Get the video's FPS (Frames Per Second) to control playback speed
    fps = cap.get(cv2.CAP_PROP_FPS)
    stframe = st.empty()  # Placeholder for the video frames

    # Use MediaPipe Pose to detect landmarks (GPU-enabled)
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("No more frames to process.")
                break

            # Convert the image color to RGB (MediaPipe expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            # Draw pose landmarks on the frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                )

            # Display the resulting frame in Streamlit (simulating a video stream)
            stframe.image(frame, channels='BGR', use_column_width=True)

            # Control the video playback speed
            # time.sleep(1.0 / fps)  # Adjust frame timing based on FPS

    cap.release()

# Streamlit app
st.title("CUDA-Accelerated Pose Landmark Detection with MediaPipe")

# File uploader for the video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.text("Processing video, please wait...")
    process_video(uploaded_file)
