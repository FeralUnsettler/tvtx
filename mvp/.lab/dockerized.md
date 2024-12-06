# To create a Dockerized Streamlit app that can be deployed on Azure Linux VMs (B1s and B2ats v2), you will need to prepare a few files: a `Dockerfile`, a `requirements.txt`, and your main Python app script. Below is an example of how to structure your project and configure it for deployment

## Step 1: Project Structure

Create a directory for your project with the following structure:

```plaintext
my_streamlit_app/
│
├── app.py                 # Your main Streamlit app script
├── requirements.txt       # Python dependencies
└── Dockerfile             # Docker configuration
```

### Step 2: `app.py`

Here’s your main Streamlit application with the webcam and video upload functionalities:

```python
import streamlit as st
import cv2
import tempfile
import mediapipe as mp
import time
import torch
import pickle
import os
import numpy as np

st.set_page_config(layout="wide")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.write(f"Using device: {device}")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

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

def process_video_from_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    stframe = st.empty()
    landmarks_data = {}
    record_landmarks = st.sidebar.checkbox("Record Pose Landmarks")

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
            time.sleep(1 / 30)

    cap.release()

    if record_landmarks:
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

st.title("Real-Time Pose Detection with Stroke Recognition and Export to Pickle")

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
```

### Step 3: `requirements.txt`

This file lists all the dependencies for your application. Here’s a basic example:

```plaintext
streamlit
opencv-python
mediapipe
torch
numpy
```

### Step 4: `Dockerfile`

Create a `Dockerfile` in the same directory to build the Docker image:

```dockerfile
# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY app.py .

# Expose the port the app runs on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 5: Build and Run Docker Container

1. **Build the Docker image**: Navigate to your project directory and run:

   ```bash
   docker build -t my_streamlit_app .
   ```

2. **Run the Docker container**:

   ```bash
   docker run -p 8501:8501 my_streamlit_app
   ```

### Step 6: Deploy to Azure

1. **Create an Azure VM**: Use the Azure portal to create a new Linux VM (choose either B1s or B2ats v2).
2. **Install Docker**: SSH into your VM and install Docker with the following commands:

   ```bash
   sudo apt-get update
   sudo apt-get install -y docker.io
   sudo systemctl start docker
   sudo systemctl enable docker
   ```

3. **Transfer your application to the VM**: You can use `scp` to copy your project folder to the VM:

   ```bash
   scp -r my_streamlit_app username@your_vm_ip:/path/to/destination
   ```

4. **Build and run your Docker container on Azure**:

   ```bash
    cd /path/to/destination/my_streamlit_app
    docker build -t


    my_streamlit_app .
    docker run -d -p 8501:8501 my_streamlit_app

   ```

5. **Access your app**: Open your web browser and go to `http://your_vm_ip:8501`.

This setup should allow you to deploy your Streamlit application on Azure using Docker, enabling webcam input and video analysis functionality with stroke detection. If you need further customization or optimizations, feel free to ask!
