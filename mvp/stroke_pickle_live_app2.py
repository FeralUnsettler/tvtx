import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import pickle
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

        # Detect Serve
        if angle < 45 and right_wrist[1] < right_shoulder[1]:
            return "Serve"
        elif angle > 100 and right_wrist[0] > right_shoulder[0]:  # Forehand
            return "Forehand"
        elif angle > 100 and right_wrist[0] < right_shoulder[0]:  # Backhand
            return "Backhand"

    return "Unknown"

# Process video and display landmarks, with the option to record landmarks
def process_video(file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()  # Streamlit placeholder for video frames

    landmarks_data = {}

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            # Record landmarks if they exist
            if results.pose_landmarks:
                landmarks = {
                    landmark_id: (landmark.x, landmark.y, landmark.z, landmark.visibility)
                    for landmark_id, landmark in enumerate(results.pose_landmarks.landmark)
                }
                stroke_type = detect_stroke(landmarks)
                
                # Draw landmarks on frame and display stroke type
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                )
                cv2.putText(frame, f"Stroke: {stroke_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                landmarks_data[frame_num] = landmarks

            stframe.image(frame, channels='BGR', use_column_width=True)
            frame_num += 1

    cap.release()
    
    # Save landmarks to a pickle file
    with open("pose_landmarks_data.pkl", "wb") as f:
        pickle.dump(landmarks_data, f)
    st.success("Landmark data has been saved.")

    # Allow download of the pickle file
    with open("pose_landmarks_data.pkl", "rb") as f:
        st.download_button(
            label="Download Pose Landmark Data",
            data=f,
            file_name="pose_landmarks_data.pkl",
            mime="application/octet-stream"
        )

# Streamlit app interface
st.title("Pose Detection with Stroke Recognition and Export to Pickle")

# Sidebar for video input options
st.sidebar.title("Input Options")
video_source = st.sidebar.radio("Choose video source:", ("Webcam", "Upload a video file"))

# Webcam or video file processing
if video_source == "Webcam":
    st.info("Please allow access to the webcam to start.")
    camera_input = st.camera_input("Record a short video")
    if camera_input:
        st.text("Processing webcam video, please wait...")
        process_video(camera_input)
else:
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    if uploaded_file:
        st.text("Processing uploaded video, please wait...")
        process_video(uploaded_file)
