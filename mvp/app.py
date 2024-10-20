import streamlit as st
import cv2
import numpy as np
import pose_detection_app as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load Google's MediaPipe Pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define a class to process video frames
class TennisPoseDetector(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose()

    def transform(self, frame):
        # Convert frame to RGB for MediaPipe Pose processing
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        # Draw pose landmarks on the image if detected
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            # Extract landmarks for service pose detection (simplified)
            landmarks = results.pose_landmarks.landmark
            shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

            # Example check for a tennis serve pose based on relative positions
            if shoulder.y > elbow.y > wrist.y:  # basic "arm up" check
                cv2.putText(frame, 'Serve Pose Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return frame

# Streamlit interface
st.title("Real-Time Tennis Service Pose Detection")
st.write("Detect tennis service poses in real time using your smartphone camera and Google's Pose API.")

# WebRTC video streamer
webrtc_streamer(key="tennis-pose-detector", video_transformer_factory=TennisPoseDetector)

# Instructions or additional information
st.markdown("""
This app uses **Google's MediaPipe Pose API** to identify tennis serve poses from a live video stream.
Simply allow access to your camera and perform a tennis serve gesture in front of the camera to see if it gets detected.
""")
