To create a shell script that sets up the entire structure for your Streamlit application project in Visual Studio Code (VSCode), you can use the following script. This script will create the necessary directories, files, and sample content needed for your project, including the Docker configuration, Streamlit app, and Terraform setup.

### **Shell Script (`setup_project.sh`)**

```bash
#!/bin/bash

# Define project name
PROJECT_NAME="streamlit_pose_landmark"

# Create project directory
mkdir -p "$PROJECT_NAME"

# Navigate into the project directory
cd "$PROJECT_NAME" || exit

# Create subdirectories
mkdir -p src data docker terraform

# Create main application file
cat <<EOL > src/app.py
import streamlit as st
import mediapipe as mp
import imageio
import numpy as np
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Streamlit title and uploader
st.title("Pose Landmark Detection with Video Upload")
video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

# Process video
if video_file is not None:
    # Load video using imageio (FFmpeg)
    video_reader = imageio.get_reader(video_file)

    # Placeholder for frames
    frame_placeholder = st.empty()

    # Process frames
    for frame_number, frame in enumerate(video_reader):
        logging.info(f"Processing frame {frame_number}")

        # Convert frame to RGB
        rgb_frame = np.array(frame)

        # Pose detection
        result = pose.process(rgb_frame)

        # Draw landmarks if detected
        if result.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                rgb_frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        # Display the frame in the Streamlit app
        frame_placeholder.image(rgb_frame, channels="RGB")

        # Sleep to simulate a 30 FPS rate
        time.sleep(1 / 30)
EOL

# Create Dockerfile
cat <<EOL > docker/Dockerfile
# Base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/ ./src

# Expose Streamlit default port
EXPOSE 8501

# Streamlit command to run the app
ENTRYPOINT ["streamlit", "run", "src/app.py"]
EOL

# Create requirements.txt
cat <<EOL > requirements.txt
streamlit==1.25.0
mediapipe==0.10.0
imageio==2.31.0
imageio-ffmpeg==0.4.8
opencv-python-headless==4.5.5.64
EOL

# Create streamlit config file
cat <<EOL > streamlit.config.toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = false
EOL

# Create Terraform files
cat <<EOL > terraform/main.tf
provider "google" {
  credentials = file("<path_to_service_account_key>.json")
  project     = "<your_project_id>"
  region      = "us-central1"
}

resource "google_compute_network" "vpc_network" {
  name = "vpc-network"
}

resource "google_compute_instance" "streamlit_instance" {
  name         = "streamlit-app"
  machine_type = "e2-micro"
  zone         = "us-central1-a"

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
      size  = 30
    }
  }

  network_interface {
    network = google_compute_network.vpc_network.name

    access_config {
      # Include an access config to give the VM a public IP address
    }
  }

  metadata_startup_script = <<-EOT
    #! /bin/bash
    sudo apt-get update
    sudo apt-get install -y docker.io
    sudo systemctl start docker
    sudo systemctl enable docker

    # Clone GitHub repo and build Docker image
    git clone https://github.com/<your_username>/<your_repo>.git
    cd <your_repo>
    sudo docker build -t streamlit-app .
    sudo docker run -d -p 80:8501 streamlit-app
  EOT
}

resource "google_compute_firewall" "default" {
  name    = "allow-http"
  network = google_compute_network.vpc_network.name

  allow {
    protocol = "tcp"
    ports    = ["80"]
  }
}
EOL

# Create .gitignore file
cat <<EOL > .gitignore
__pycache__/
*.pyc
.env
.DS_Store
venv/
EOL

# Print completion message
echo "Project structure for $PROJECT_NAME has been created successfully."
```

### **How to Use the Script**

1. **Create the Script:**
   - Open your terminal and create a new file named `setup_project.sh`.
   - Copy and paste the above script into `setup_project.sh`.

2. **Make the Script Executable:**
   ```bash
   chmod +x setup_project.sh
   ```

3. **Run the Script:**
   ```bash
   ./setup_project.sh
   ```

4. **Customize:**
   - Update `<path_to_service_account_key>.json`, `<your_project_id>`, `<your_username>`, and `<your_repo>` in the `main.tf` file as needed.

5. **Open in VSCode:**
   - Open VSCode and load your project directory. The structure should be ready for you to start developing your application.

This script sets up a complete project structure for your Streamlit app with pose landmark detection, Docker configuration, and Terraform deployment configuration, making it easy to manage your project in VSCode.