import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import numpy as np

# Set page configuration
st.set_page_config(layout="wide")

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

        # Detect Serve, Forehand, or Backhand based on angle and position
        if angle < 45 and right_wrist[1] < right_shoulder[1]:
            return "Serve"
        elif angle > 100 and right_wrist[0] > right_shoulder[0]:  # Forehand
            return "Forehand"
        elif angle > 100 and right_wrist[0] < right_shoulder[0]:  # Backhand
            return "Backhand"

    return "Unknown"

# Process video frames and display with pose landmarks and stroke type
def process_camera_frame(frame):
    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Record and display landmarks if detected
    if results.pose_landmarks:
        landmarks = {
            landmark_id: (landmark.x, landmark.y, landmark.z, landmark.visibility)
            for landmark_id, landmark in enumerate(results.pose_landmarks.landmark)
        }
        stroke_type = detect_stroke(landmarks)
        
        # Draw landmarks on frame
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
        )
        # Display stroke type on the frame
        cv2.putText(frame, f"Stroke: {stroke_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame

# Streamlit app interface
st.title("Real-Time Pose Detection with Stroke Recognition")

# Initialize MediaPipe Pose model
with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
    # Sidebar for input options
    st.sidebar.title("Input Options")
    video_source = st.sidebar.radio("Choose video source:", ("Webcam", "Upload a video file"))

    # Webcam input handling
    if video_source == "Webcam":
        st.info("Please allow access to the webcam.")
        camera_input = st.camera_input("Record from webcam")
        if camera_input is not None:
            # Read the frame from camera input
            temp_file = tempfile.NamedTemporaryFile(delete=False)  # Save the image temporarily
            temp_file.write(camera_input.getvalue())  # Get bytes and write to file
            frame = cv2.imread(temp_file.name)  # Read file as an image with OpenCV
            
            # Process the frame for pose detection and stroke recognition
            processed_frame = process_camera_frame(frame)
            
            # Display the processed frame with annotations
            st.image(processed_frame, channels="BGR", use_column_width=True)

    # Uploaded video file handling
    else:
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)

            stframe = st.empty()  # Placeholder for video frames

            # Process and display each frame in the video file
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = process_camera_frame(frame)
                stframe.image(processed_frame, channels="BGR", use_column_width=True)

            cap.release()
