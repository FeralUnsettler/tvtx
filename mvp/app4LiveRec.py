import streamlit as st
import cv2
import tempfile
import mediapipe as mp
import time
import torch
import pickle
import numpy as np
import os

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.sidebar.write(f"Using device: {device}")

# Initialize MediaPipe Pose and Drawing Utils
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Helper function to detect stroke type based on key landmarks
def detect_stroke(landmarks):
    if landmarks:
        right_shoulder = landmarks[12]
        right_elbow = landmarks[14]
        right_wrist = landmarks[16]
        
        shoulder_to_elbow = np.array([right_elbow[0] - right_shoulder[0], right_elbow[1] - right_shoulder[1]])
        elbow_to_wrist = np.array([right_wrist[0] - right_elbow[0], right_wrist[1] - right_elbow[1]])
        
        angle = np.degrees(np.arccos(
            np.dot(shoulder_to_elbow, elbow_to_wrist) / (np.linalg.norm(shoulder_to_elbow) * np.linalg.norm(elbow_to_wrist))
        ))

        if angle < 45 and right_wrist[1] < right_shoulder[1]:
            return "Serve"
        elif angle > 100 and right_wrist[0] > right_shoulder[0]:
            return "Forehand"
        elif angle > 100 and right_wrist[0] < right_shoulder[0]:
            return "Backhand"

    return "Unknown"

# Function to process video, detect strokes, and save landmarks to a pickle file
def process_video(cap, record=False):
    fps = 30  # Set target frame rate for playback
    stframe = st.empty()  # Placeholder for video frames
    landmarks_data = {}

    # Prepare to record video if needed
    video_writer = None
    if record:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter("recorded_video.avi", fourcc, fps, (640, 480))

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
        frame_num = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (record and time.time() - start_time > 20):
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = {
                    landmark_id: (landmark.x, landmark.y, landmark.z, landmark.visibility)
                    for landmark_id, landmark in enumerate(results.pose_landmarks.landmark)
                }
                landmarks_data[frame_num] = landmarks

                stroke_type = detect_stroke(landmarks)
                
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                )
                cv2.putText(frame, f"Stroke: {stroke_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            stframe.image(frame, channels='BGR', use_column_width=True)
            if record:
                video_writer.write(frame)
            frame_num += 1
            time.sleep(1.0 / fps)

    cap.release()
    if video_writer:
        video_writer.release()
    
    with open("pose_landmarks_data.pkl", "wb") as f:
        pickle.dump(landmarks_data, f)
    st.sidebar.success("Landmark data has been saved.")

    if record:
        with open("recorded_video.avi", "rb") as f:
            st.sidebar.download_button(
                label="Download 20-second Recorded Video",
                data=f,
                file_name="recorded_video.avi",
                mime="video/x-msvideo"
            )

    with open("pose_landmarks_data.pkl", "rb") as f:
        st.sidebar.download_button(
            label="Download Pose Landmark Data",
            data=f,
            file_name="pose_landmarks_data.pkl",
            mime="application/octet-stream"
        )

    os.remove("pose_landmarks_data.pkl")

# Streamlit app interface
st.title("TVTxMindVision - Pose Detection & Stroke Recognition")

# Instructions for the user
st.markdown("""
### Instructions:
1. **Select Video Source**: Use the sidebar to choose between uploading a video or using the live webcam feed.
2. **For Uploaded Videos**: Simply upload the video file, and it will be processed automatically.
3. **For Webcam Recording**: Click "Start Live Webcam Recording" to record a 20-second video using your webcam.
4. **Stroke Detection**: The app will detect and display the stroke type (Serve, Forehand, Backhand) in real-time.
5. **Download Options**: After processing, download the landmark data and recorded video (if applicable) from the sidebar.
""")

# Sidebar for source selection
source = st.sidebar.selectbox("Select Video Source", ("Upload a video file", "Live Webcam"))

if source == "Upload a video file":
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()
        
        cap = cv2.VideoCapture(tfile.name)
        st.text("Processing uploaded video, please wait...")
        process_video(cap)
        
else:
    if st.sidebar.button("Start Live Webcam Recording (20 seconds)"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam.")
        else:
            st.text("Processing live webcam feed, please wait...")
            process_video(cap, record=True)
