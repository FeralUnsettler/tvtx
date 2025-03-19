import streamlit as st
import cv2
import numpy as np
import time
import pickle
from collections import Counter
from hailo_platform import hailo_infer  # Biblioteca para inferência Hailo
from picam_driver import PiCamGS  # Biblioteca para Inno PiCam GS

# Configurações globais
FRAME_RESOLUTION = (640, 480)
FRAME_RATE = 30
MODEL_PATH = 'pose_estimation.hailo'

# Inicializar a câmera e o modelo
def initialize_camera():
    return PiCamGS(resolution=FRAME_RESOLUTION, framerate=FRAME_RATE)

def initialize_model():
    return hailo_infer.load_model(MODEL_PATH)

# Função para detectar golpes com base nos landmarks
def detect_stroke(landmarks):
    if landmarks:
        right_shoulder, right_elbow, right_wrist = landmarks[12], landmarks[14], landmarks[16]
        shoulder_to_elbow = np.array([right_elbow[0] - right_shoulder[0], right_elbow[1] - right_shoulder[1]])
        elbow_to_wrist = np.array([right_wrist[0] - right_elbow[0], right_wrist[1] - right_elbow[1]])
        
        angle = np.degrees(np.arccos(
            np.dot(shoulder_to_elbow, elbow_to_wrist) / 
            (np.linalg.norm(shoulder_to_elbow) * np.linalg.norm(elbow_to_wrist) + 1e-6)
        ))

        if angle < 45 and right_wrist[1] < right_shoulder[1]:
            return "Serve"
        elif angle > 100 and right_wrist[0] > right_shoulder[0]:
            return "Forehand"
        elif angle > 100 and right_wrist[0] < right_shoulder[0]:
            return "Backhand"

    return "Unknown"

# Função para processar vídeo ao vivo
def process_video(camera, model, duration=20):
    stroke_counter = Counter()
    stframe = st.image([])
    stats_frame = st.sidebar.empty()
    
    start_time = time.time()
    frames = []

    while time.time() - start_time < duration:
        frame = camera.capture_frame()
        landmarks = model.infer(frame)

        stroke_type = detect_stroke(landmarks)
        stroke_counter[stroke_type] += 1

        cv2.putText(frame, f"Stroke: {stroke_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        stframe.image(frame, channels="BGR")
        stats_frame.text(str(stroke_counter))
        frames.append(frame)
    
    return frames, stroke_counter

# Função para salvar vídeo e dados
def save_results(frames, stroke_counter):
    video_output = "processed_video.mp4"
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

    pickle_output = "stroke_data.pkl"
    with open(pickle_output, 'wb') as f:
        pickle.dump(stroke_counter, f)
    
    return video_output, pickle_output

# Interface Streamlit
def main():
    st.title("🎾 Análise de Golpes no Tênis - Raspberry Pi 5 & Hailo 8L")
    st.markdown("""
    ### Passos para Configuração:
    1. Conecte a Inno PiCam GS ao Raspberry Pi 5.
    2. Instale os drivers da câmera: `sudo apt install picam-gs-drivers`
    3. Configure o Hailo 8L com `hailo-setup`
    4. Clone este repositório e instale as dependências: `pip install -r requirements.txt`
    5. Execute a aplicação com `streamlit run app.py`
    """)

    if st.button("Iniciar Gravação"):
        st.info("Inicializando câmera e modelo...")
        camera = initialize_camera()
        model = initialize_model()

        st.success("Câmera e modelo inicializados. Gravando...")
        frames, stroke_counter = process_video(camera, model)

        st.info("Processando resultados...")
        video_output, pickle_output = save_results(frames, stroke_counter)

        st.sidebar.download_button("Baixar Vídeo", open(video_output, "rb"), file_name="processed_video.mp4")
        st.sidebar.download_button("Baixar Dados", open(pickle_output, "rb"), file_name="stroke_data.pkl")
        st.success("Gravação concluída!")

if __name__ == "__main__":
    main()