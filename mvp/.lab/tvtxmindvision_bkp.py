import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import mediapipe as mp
import numpy as np
import cv2
from collections import Counter
import pickle
from datetime import datetime

# Setup MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# WebRTC Configuration
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Constants
FRAME_WIDTH, FRAME_HEIGHT = 800, 600
FRAME_SKIP = 2

# Stroke counter
stroke_counter = Counter()

# Function to detect stroke type
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

# Custom VideoProcessor for Streamlit-WebRTC
class StrokeDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.stroke_counter = Counter()
        self.frame_count = 0
        self.start_time = datetime.now()
        self.recording_frames = []

    def recv(self, frame):
        self.frame_count += 1

        # Skip frames for performance
        if self.frame_count % FRAME_SKIP != 0:
            return frame

        # Convert frame to RGB and resize to 800x600
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process frame with MediaPipe
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            stroke_type = detect_stroke(landmarks)
            self.stroke_counter[stroke_type] += 1

            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )
            # Display stroke type
            cv2.putText(img, f"Stroke: {stroke_type}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Record video
        self.recording_frames.append(img)

        # Limit recording to 20 seconds
        if (datetime.now() - self.start_time).seconds > 20:
            self.recording_frames = self.recording_frames[-20 * 30:]  # Keep only the last 20 seconds of frames

        return frame.from_ndarray(img, format="bgr24")

    def save_recording(self):
        # Save the recording to a file
        if self.recording_frames:
            out_filename = f"stroke_detection_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_filename, fourcc, 30, (FRAME_WIDTH, FRAME_HEIGHT))

            for frame in self.recording_frames:
                out.write(frame)

            out.release()
            return out_filename

        return None

# Streamlit UI
st.markdown(
    """
    <style>
        body {
            background-image: url('media/luxx/Crucial_500GB/Media/Videos/Captures/tvtx/img/1b.webp');
            background-size: cover;
        }
    </style>
    """, unsafe_allow_html=True
)

st.title("🎾 BMDS®MindVision")
st.markdown("Detect tennis strokes in real-time using GPU acceleration!")

webrtc_ctx = webrtc_streamer(
    key="stroke-detection",
    mode="video",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=StrokeDetectionProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": FRAME_WIDTH},
            "height": {"ideal": FRAME_HEIGHT},
            "frameRate": {"ideal": 30},
        },
        "audio": False,
    },
)

if webrtc_ctx.video_processor:
    processor = webrtc_ctx.video_processor
    stroke_counter = processor.stroke_counter
    total_strokes = sum(stroke_counter.values())

    if total_strokes > 0:
        st.sidebar.title("Stroke Statistics")
        st.sidebar.write("### Counts")
        for stroke, count in stroke_counter.items():
            st.sidebar.write(f"{stroke}: {count} ({(count / total_strokes) * 100:.1f}%)")

        st.sidebar.bar_chart({stroke: count / total_strokes for stroke, count in stroke_counter.items()})

        # Save recording button
        if st.sidebar.button("Save Recording"):
            filename = processor.save_recording()
            if filename:
                st.sidebar.success(f"Recording saved as {filename}")
