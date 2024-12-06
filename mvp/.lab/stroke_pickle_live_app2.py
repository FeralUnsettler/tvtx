import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image
import tempfile

# Initialize MediaPipe Pose and Drawing Utils
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Set Streamlit page configuration
st.set_page_config(layout="wide")

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

        # Detect Serve, Forehand, or Backhand based on angle and position
        if angle < 45 and right_wrist[1] < right_shoulder[1]:
            return "Serve"
        elif angle > 100 and right_wrist[0] > right_shoulder[0]:  # Forehand
            return "Forehand"
        elif angle > 100 and right_wrist[0] < right_shoulder[0]:  # Backhand
            return "Backhand"  # Ensure "Backhand" is returned here

    return "Unknown"

# Function to process frames and add pose landmarks
def process_frame(image, pose):
    # Convert image to RGB (MediaPipe expects RGB)
    rgb_image = np.array(image.convert("RGB"))
    results = pose.process(rgb_image)

    # Draw landmarks on the image if detected
    if results.pose_landmarks:
        annotated_image = rgb_image.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
        )
        landmarks = [
            (landmark.x, landmark.y, landmark.z, landmark.visibility)
            for landmark in results.pose_landmarks.landmark
        ]
        stroke_type = detect_stroke(landmarks)
        return Image.fromarray(annotated_image), stroke_type

    return image, "Unknown"

# Streamlit app interface
st.title("Real-Time Pose Detection with Stroke Recognition")

# Sidebar for input options
st.sidebar.title("Input Options")
video_source = st.sidebar.radio("Choose video source:", ("Webcam", "Upload a video file"))

# Initialize MediaPipe Pose model
with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    # Webcam input handling
    if video_source == "Webcam":
        st.info("Please allow access to the webcam.")
        camera_input = st.camera_input("Record from webcam")

        if camera_input is not None:
            # Read the frame from camera input
            img = Image.open(camera_input)

            # Process the frame for pose detection and stroke recognition
            processed_img, stroke_type = process_frame(img, pose)
            
            # Display the processed frame with annotations
            st.image(processed_img, use_column_width=True)
            st.write(f"Detected Stroke Type: {stroke_type}")

    # Uploaded video file handling
    else:
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            st.video(tfile.name)
            st.write("Note: Real-time pose detection on uploaded video files may not work with Streamlit's video player.")
