import streamlit as st
import cv2
import tempfile
import mediapipe as mp
import time
import torch
import pickle
import os
import numpy as np

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.write(f"Using device: {device}")

# Initialize MediaPipe Pose and Drawing Utils
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Helper function to detect stroke type based on key landmarks
def detect_stroke(landmarks):
    if landmarks:
        right_shoulder = landmarks[12]
        right_elbow = landmarks[14]
        right_wrist = landmarks[16]
        
        # Calculate angle at the right elbow
        shoulder_to_elbow = np.array([right_elbow[0] - right_shoulder[0], right_elbow[1] - right_shoulder[1]])
        elbow_to_wrist = np.array([right_wrist[0] - right_elbow[0], right_wrist[1] - right_elbow[1]])
        
        angle = np.degrees(np.arccos(
            np.dot(shoulder_to_elbow, elbow_to_wrist) / (np.linalg.norm(shoulder_to_elbow) * np.linalg.norm(elbow_to_wrist))
        ))

        # Detect Serve
        if angle < 45 and right_wrist[1] < right_shoulder[1]:
            return "Serve"
        elif angle > 100 and right_wrist[0] > right_shoulder[0]:  # Forehand
            return "Forehand"
        elif angle > 100 and right_wrist[0] < right_shoulder[0]:  # Backhand
            return "Backhand"

    return "Unknown"

# Function to process live webcam feed
def process_video_from_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set to 30 FPS for real-time processing
    stframe = st.empty()  # Placeholder for video frames

    landmarks_data = {}
    record_landmarks = st.checkbox("Record Pose Landmarks")

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Could not access webcam.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = {
                    landmark_id: (landmark.x, landmark.y, landmark.z, landmark.visibility)
                    for landmark_id, landmark in enumerate(results.pose_landmarks.landmark)
                }
                stroke_type = detect_stroke(landmarks)
                
                # Display landmarks and stroke type on frame
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                )
                cv2.putText(frame, f"Stroke: {stroke_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if record_landmarks:
                    landmarks_data[frame_num] = landmarks

            stframe.image(frame, channels='BGR', use_column_width=True)
            frame_num += 1
            time.sleep(1 / 30)  # 30 FPS

    cap.release()

    # Save landmarks data to a pickle file if recording is enabled
    if record_landmarks:
        with open("pose_landmarks_data.pkl", "wb") as f:
            pickle.dump(landmarks_data, f)
        st.success("Landmark data has been saved.")

        # Provide download option for the pickle file
        with open("pose_landmarks_data.pkl", "rb") as f:
            st.download_button(
                label="Download Pose Landmark Data",
                data=f,
                file_name="pose_landmarks_data.pkl",
                mime="application/octet-stream"
            )
        os.remove("pose_landmarks_data.pkl")

# Function to process uploaded video
def process_uploaded_video(file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    fps = 30
    stframe = st.empty()

    landmarks_data = {}

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("End of video.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = {
                    landmark_id: (landmark.x, landmark.y, landmark.z, landmark.visibility)
                    for landmark_id, landmark in enumerate(results.pose_landmarks.landmark)
                }
                stroke_type = detect_stroke(landmarks)
                
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                )
                cv2.putText(frame, f"Stroke: {stroke_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                landmarks_data[frame_num] = landmarks

            stframe.image(frame, channels='BGR', use_column_width=True)
            frame_num += 1
            time.sleep(1 / fps)

    cap.release()
    
    with open("pose_landmarks_data.pkl", "wb") as f:
        pickle.dump(landmarks_data, f)
    st.success("Landmark data has been saved.")

    with open("pose_landmarks_data.pkl", "rb") as f:
        st.download_button(
            label="Download Pose Landmark Data",
            data=f,
            file_name="pose_landmarks_data.pkl",
            mime="application/octet-stream"
        )
    os.remove("pose_landmarks_data.pkl")

# Streamlit app interface
st.title("Real-Time Pose Detection with Stroke Recognition and Export to Pickle")

# Video input selection
st.sidebar.title("Input Options")
video_source = st.sidebar.radio("Choose video source:", ("Webcam", "Upload a video file"))

if video_source == "Webcam":
    st.text("Analyzing live video from webcam...")
    process_video_from_camera()
else:
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        st.text("Processing uploaded video, please wait...")
        process_uploaded_video(uploaded_file)
