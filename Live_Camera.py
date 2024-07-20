import streamlit as st
import cv2
from PIL import Image
import numpy as np
import base64
import tempfile
import os

# Function to perform road damage detection on an image
def detect_damage_in_image(image):
    # Placeholder for YOLO v8 model inference
    gray_image = np.array(image.convert('L'))
    return gray_image


# Function to perform road damage detection on a video
def detect_damage_in_video(video_file):
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, "temp_video.mp4")

    # Save the uploaded video to the temporary directory
    with open(temp_file_path, "wb") as f:
        f.write(video_file.read())

    # Open the video using OpenCV VideoCapture
    cap = cv2.VideoCapture(temp_file_path)

    return cap
# Function to perform road damage detection on live camera feed
def detect_damage_in_live_camera():
    # Placeholder for YOLO v8 model inference
    cap = cv2.VideoCapture(0)
    return cap

# Function to convert video frames to base64 string
def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode()

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f4f4f9;
        color: #333;
        font-family: 'Arial', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #4b6cb7;
        margin-bottom: 20px;
    }
    .header {
        font-size: 24px;
        font-weight: bold;
        color: #4b6cb7;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .frame {
        border: 2px solid #4b6cb7;
        border-radius: 10px;
        padding: 10px;
        background-color: #ffffff;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .button {
        text-align: center;
        margin-top: 20px;
    }
    .btn-primary {
        background-color: #4b6cb7;
        color: #ffffff;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        border-radius: 5px;
    }
    .btn-primary:hover {
        background-color: #3a57a0;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown('<div class="title">Road Damage Detection</div>', unsafe_allow_html=True)

# Option 1: Upload Photo
st.markdown('<div class="header">Upload Photo</div>', unsafe_allow_html=True)
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.markdown('<div class="frame">', unsafe_allow_html=True)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    if st.button("Detect Damage in Image"):
        result_image = detect_damage_in_image(image)
        st.markdown('<div class="frame">', unsafe_allow_html=True)
        st.image(result_image, caption="Detected Damage", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Option 2: Upload Video
st.markdown('<div class="header">Upload Video</div>', unsafe_allow_html=True)
uploaded_video = st.file_uploader("Upload a video", type=["mp4"])
if uploaded_video is not None:
    st.markdown('<div class="frame">', unsafe_allow_html=True)
    st.video(uploaded_video)
    st.markdown('</div>', unsafe_allow_html=True)
    if st.button("Detect Damage in Video"):
        video_file = detect_damage_in_video(uploaded_video)
        st.markdown('<div class="frame">', unsafe_allow_html=True)
        while True:
            ret, frame = video_file.read()
            if not ret:
                break
            st.image(frame, channels="BGR", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Option 3: Live Camera
st.markdown('<div class="header">Live Camera</div>', unsafe_allow_html=True)
if st.button("Start Live Camera"):
    live_camera = detect_damage_in_live_camera()
    video_placeholder = st.empty()
    st.markdown('<div class="frame">', unsafe_allow_html=True)
    while live_camera.isOpened():
        ret, frame = live_camera.read()
        if not ret:
            st.write("Failed to grab frame")
            break
        # Placeholder for YOLO v8 model inference
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        video_placeholder.image(gray_frame, channels="GRAY")
    live_camera.release()
    st.markdown('</div>', unsafe_allow_html=True)

# Styling the detect buttons
st.markdown('<style>.stButton button { background-color: #4b6cb7; color: white; }</style>', unsafe_allow_html=True)
