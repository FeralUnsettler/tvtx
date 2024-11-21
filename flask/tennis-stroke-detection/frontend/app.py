import streamlit as st
import requests
import json

API_URL = "http://backend:5000"

st.title("Tennis Stroke Detection")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_file:
    st.text("Processing video...")
    features = [0.5, 0.2, -0.1, 0.8, -0.2, 0.4]  # Dummy features
    handedness = st.selectbox("Select handedness", ["Right", "Left"])

    response = requests.post(f"{API_URL}/predict", json={"features": features, "handedness": handedness})
    if response.status_code == 200:
        stroke_type = response.json()["stroke_type"]
        st.write(f"Predicted Stroke Type: {stroke_type}")

        if st.button("Save Stroke"):
            requests.post(f"{API_URL}/record", json={"features": features, "handedness": handedness, "stroke_type": stroke_type})
            st.write("Stroke data saved!")
