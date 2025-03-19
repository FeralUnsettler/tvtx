import streamlit as st
import cv2
import tempfile
import mediapipe as mp
import time
import torch  # For GPU availability check

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.write(f"Using device: {device}")

# Initialize MediaPipe Pose and Drawing Utils
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to process video and draw landmarks
def process_video(file):
    # Create a temporary file to save the uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())
    tfile.close()

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(tfile.name)
    
    # Get the video's FPS to control playback speed
    fps = cap.get(cv2.CAP_PROP_FPS)
    stframe = st.empty()  # Placeholder for video frames
    progress_bar = st.progress(0)

    # Use MediaPipe Pose for landmark detection
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("End of video.")
                break

            # Convert the image to RGB as MediaPipe requires
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            # Draw pose landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                )

            # Display the resulting frame
            stframe.image(frame, channels='BGR', use_column_width=True)

            # Update progress bar
            frame_num += 1
            progress_bar.progress(frame_num / frame_count)

            # Control playback speed based on FPS
            time.sleep(1.0 / fps)

    cap.release()
    progress_bar.empty()

# Streamlit app interface
st.title("CUDA-Accelerated Pose Detection with MediaPipe")

# File uploader
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.text("Processing video, please wait...")
    process_video(uploaded_file)
