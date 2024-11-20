import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import mediapipe as mp
import numpy as np
import cv2
import pickle
import tempfile
import os

# Initialize MediaPipe Pose and Drawing Utils
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to detect the type of tennis stroke
def detect_stroke(landmarks):
    if landmarks:
        right_shoulder = landmarks[12]
        right_elbow = landmarks[14]
        right_wrist = landmarks[16]

        # Compute angle between shoulder, elbow, and wrist
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

# WebRTC Transformer for video processing
class PoseVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
            landmarks = [
                (lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark
            ]
            stroke_type = detect_stroke(landmarks)
            cv2.putText(img, f"Stroke: {stroke_type}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return img

# Streamlit App
st.title("Tennis Pose Detection with WebRTC")
st.sidebar.title("Controls")

# Instructions
st.markdown("""
### Instructions:
1. **Start the Camera**: Click on "Start Video" to use your device's camera.
2. **Real-time Detection**: View real-time pose detection and stroke recognition.
3. **Download Pose Data**: Option to save the processed data locally.
""")

# WebRTC Video Stream
ctx = webrtc_streamer(
    key="pose-detection",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=PoseVideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Download Options
if st.sidebar.button("Download Pose Data"):
    pose_data = {"landmarks": "Sample pose data for demo"}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        pickle.dump(pose_data, f)
        st.sidebar.download_button(
            label="Download Pose Data",
            data=f,
            file_name="pose_data.pkl",
            mime="application/octet-stream"
        )
        os.unlink(f.name)
