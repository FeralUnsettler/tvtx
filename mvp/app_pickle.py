import streamlit as st
import cv2
import tempfile
import mediapipe as mp
import time
import torch
import pickle
import os

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.write(f"Using device: {device}")

# Initialize MediaPipe Pose and Drawing Utils
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to process video, display frames, and save landmarks to a pickle file
def process_video(file):
    # Create a temporary file to save the uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())
    tfile.close()

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    stframe = st.empty()  # Placeholder for video frames

    # Dictionary to store landmarks data
    landmarks_data = {}

    # Use MediaPipe Pose for landmark detection
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
        frame_num = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("End of video.")
                break

            # Convert the frame to RGB as MediaPipe requires
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            # Extract pose landmarks data
            if results.pose_landmarks:
                landmarks = {
                    landmark_id: (landmark.x, landmark.y, landmark.z, landmark.visibility)
                    for landmark_id, landmark in enumerate(results.pose_landmarks.landmark)
                }
                landmarks_data[frame_num] = landmarks

                # Draw landmarks on the frame for visualization
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                )

            # Display the resulting frame in Streamlit (simulating video preview)
            stframe.image(frame, channels='BGR', use_column_width=True)

            frame_num += 1
            time.sleep(1.0 / fps)  # Control playback speed

    cap.release()
    
    # Save landmarks data to a pickle file
    pickle_file_path = "pose_landmarks_data.pkl"
    with open(pickle_file_path, "wb") as f:
        pickle.dump(landmarks_data, f)
    st.success("Landmark data has been saved.")

    # Provide download option for the pickle file
    with open(pickle_file_path, "rb") as f:
        st.download_button(
            label="Download Pose Landmark Data",
            data=f,
            file_name="pose_landmarks_data.pkl",
            mime="application/octet-stream"
        )

    # Remove the temporary pickle file after download to clean up
    os.remove(pickle_file_path)

# Streamlit app interface
st.title("CUDA-Accelerated Pose Detection with MediaPipe and Export to Pickle")

# File uploader
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.text("Processing video, please wait...")
    process_video(uploaded_file)
