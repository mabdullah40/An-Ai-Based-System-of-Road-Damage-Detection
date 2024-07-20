import streamlit as st
import cv2
from PIL import Image
import numpy as np
import base64
from ultralytics import YOLO
import cvzone
from sort import Sort
import math
import tempfile
import os
import matplotlib.pyplot as plt
from fpdf import FPDF
import time
from datetime import datetime

# Load YOLO model
model = YOLO('best.pt')
class_names = list(model.names.values())

# Function to perform road damage detection on an image
def detect_damage_in_image(image):
    image_np = np.array(image)
    results = model(image_np)
    damage_counts = {cls_name: 0 for cls_name in class_names}
    damage_trackers = {cls_name: {} for cls_name in class_names}
    damage_threshold = 1

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            current_class = class_names[cls]

            if conf > 0.2:
                cvzone.cornerRect(image_np, (x1, y1, w, h))
                cvzone.putTextRect(image_np, f'{current_class} {conf}', (max(0, x1), max(35, y1 - 15)), scale=1, thickness=1)
                # Update damage trackers
                if (x1, y1, x2, y2) not in damage_trackers[current_class]:
                    damage_trackers[current_class][(x1, y1, x2, y2)] = 1
                else:
                    damage_trackers[current_class][(x1, y1, x2, y2)] += 1

    # Update damage counts
    for cls_name in damage_trackers:
        for bbox, count in damage_trackers[cls_name].items():
            if count >= damage_threshold:
                damage_counts[cls_name] += 1

    generate_pdf(damage_counts)
    return image_np
# Function to perform road damage detection on a video
def detect_damage_in_video(video_file):
    damage_counts=None
    temp_dir = tempfile.mkdtemp()
    input_video_path = os.path.join(temp_dir, "input_video.mp4")
    output_video_path = os.path.join(temp_dir, "output_video.mp4")

    # Save the uploaded video to the temporary directory
    with open(input_video_path, "wb") as f:
        f.write(video_file.read())

    # Open the video using OpenCV VideoCapture
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    tracker = Sort(max_age=20, min_hits=10, iou_threshold=0.4)
    damage_counts = {cls_name: 0 for cls_name in class_names}
    damage_trackers = {cls_name: {} for cls_name in class_names}
    damage_threshold = 5

    while True:
        success, img = cap.read()
        if not success:
            break
        results = model(img, stream=True)
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                current_class = class_names[cls]

                if conf > 0.2:
                    cvzone.cornerRect(img, (x1, y1, w, h))
                    cvzone.putTextRect(img, f'{current_class} {conf}', (max(0, x1), max(35, y1 - 15)), scale=1, thickness=1)
                    current_array = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, current_array))

                    if current_class in damage_trackers:
                        if (x1, y1, x2, y2) in damage_trackers[current_class]:
                            damage_trackers[current_class][(x1, y1, x2, y2)] += 1
                        else:
                            damage_trackers[current_class][(x1, y1, x2, y2)] = 1

        results = tracker.update(detections)
        for result in results:
            x1, y1, x2, y2, _ = result

        for cls_name in damage_trackers:
            for bbox, count in damage_trackers[cls_name].items():
                if count >= damage_threshold:
                    damage_counts[cls_name] += 1
            damage_trackers[cls_name] = {bbox: count for bbox, count in damage_trackers[cls_name].items() if count < damage_threshold}

        frame_base64 = frame_to_base64(img)
        video_placeholder.image(f"data:image/jpeg;base64,{frame_base64}")

    cap.release()
    out.release()
    video_placeholder.empty()
    generate_pdf(damage_counts)



def generate_pdf(damage_counts):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 10)
            self.cell(0, 10, 'Road Damage Detection Report', 0, 1, 'C')
            self.cell(0, 10, 'Using YOLO Model', 0, 1, 'C')
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

        def chapter_title(self, title):
            self.set_font('Arial', 'B', 10)
            self.cell(0, 10, title, 0, 1, 'L')
            self.ln(4)

        def chapter_body(self, body):
            self.set_font('Arial', '', 10)
            self.multi_cell(0, 5, body)
            self.ln()

        def add_table(self, damage_counts):
            self.set_font('Arial', 'B', 10)
            self.cell(100, 10, 'Type of Damage', 1, 0, 'C', 1)
            self.cell(40, 10, 'Count', 1, 1, 'C', 1)
            self.set_font('Arial', '', 10)
            for key, value in damage_counts.items():
                self.cell(100, 10, key, 1)
                self.cell(40, 10, str(value), 1, 1)

    pdf = PDF()
    pdf.add_page()

    pdf.chapter_title('Introduction')
    intro_text = (
        "This report provides an analysis of road damages detected using the YOLO model. "
        "The primary objective is to identify and categorize different types of road damages to assist in maintenance planning. "
        "The YOLO (You Only Look Once) model is a state-of-the-art object detection system that is fast and accurate. "
        "It processes images in real-time and can identify various types of objects, including different categories of road damages. "
        "By utilizing this model, we aim to streamline the process of road inspection and maintenance scheduling.\n\n"
    )
    pdf.chapter_body(intro_text)

    pdf.chapter_title('Types of Damages')
    pdf.add_table(damage_counts)

    pdf.chapter_title('\nConclusion')
    if sum(damage_counts.values()) > 0:
        big_damage = max(damage_counts, key=damage_counts.get)
        conclusion_text = (
            f"The analysis indicates that the most common type of damage is {big_damage}. "
            "Immediate attention is recommended for areas with severe damages. Regular monitoring and timely repairs can help prevent further deterioration of the road infrastructure.\n"
        )
        pdf.chapter_body(conclusion_text)

        pdf.chapter_title('Recommendations')
        recommendations_text = (
            f"1. Prioritize repair for {big_damage}.\n"
            "2. Conduct regular monitoring to track changes and new damages.\n"
            "3. Implement a maintenance schedule based on the severity and frequency of the detected damages.\n"
            "4. Utilize advanced detection models like YOLO for continuous and efficient road inspection.\n\n"
        )
        pdf.chapter_body(recommendations_text)
    else:
        conclusion_text = (
            "The analysis did not detect any road damages. The road conditions appear to be satisfactory. "
            "Regular monitoring is still recommended to ensure the road remains in good condition and to quickly identify any future damages that may occur.\n"
        )
        pdf.chapter_body(conclusion_text)

        pdf.chapter_title('Recommendations')
        recommendations_text = (
            "1. Continue regular road inspections to maintain safety and quality.\n"
            "2. Ensure timely repairs if any damages are detected in future inspections.\n"
            "3. Utilize advanced detection models like YOLO for continuous and efficient road inspection.\n\n"
        )
        pdf.chapter_body(recommendations_text)


    pdf_file_name = 'Road_Damage_Detection_Report.pdf'
    pdf.output(pdf_file_name)

    with open(pdf_file_name, "rb") as pdf_file:
        PDFbyte = pdf_file.read()

    return pdf_file_name, PDFbyte

def detect_damage_in_live_camera():
    cap = cv2.VideoCapture(0)
    cap.set(3, 720)
    cap.set(4, 720)
    tracker = Sort(max_age=20, min_hits=5, iou_threshold=0.4)
    damage_counts = {cls_name: 0 for cls_name in class_names}
    damage_trackers = {cls_name: {} for cls_name in class_names}
    damage_threshold = 1
    video_placeholder = st.empty()

    # Create a button with the text "Click me!"
    stop_button = st.button("Stop Loop")
    
    running = True
    while cap.isOpened() and running:
        success, img = cap.read()
        if not success:
            break
        if stop_button:
            st.write("Loop stopped!")
            running = False
        results = model(img, stream=True)
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                current_class = class_names[cls]

                if conf > 0.2:
                    cvzone.cornerRect(img, (x1, y1, w, h))
                    cvzone.putTextRect(img, f'{current_class} {conf}', (max(0, x1), max(35, y1 - 15)), scale=1, thickness=1)
                    current_array = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, current_array))

                    if current_class in damage_trackers:
                        if (x1, y1, x2, y2) in damage_trackers[current_class]:
                            damage_trackers[current_class][(x1, y1, x2, y2)] += 1
                        else:
                            damage_trackers[current_class][(x1, y1, x2, y2)] = 1

        results = tracker.update(detections)
        for result in results:
            x1, y1, x2, y2, _ = result
        for cls_name in damage_trackers:
            for bbox, count in damage_trackers[cls_name].items():
                if count >= damage_threshold:
                    damage_counts[cls_name] += 1
            damage_trackers[cls_name] = {bbox: count for bbox, count in damage_trackers[cls_name].items() if count < damage_threshold}

        frame_base64 = frame_to_base64(img)
        video_placeholder.image(f"data:image/jpeg;base64,{frame_base64}")
        generate_pdf(damage_counts) 
    cap.release()
    
def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()
    frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
    return frame_base64
# Custom CSS for better styling
custom_css = """
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
"""

# Streamlit UI
st.markdown(custom_css, unsafe_allow_html=True)
st.markdown('<div class="title">Road Damage Detection</div>', unsafe_allow_html=True)

# Option 1: Upload Photo
st.markdown('<div class="header">Upload Photo</div>', unsafe_allow_html=True)
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    result_image = detect_damage_in_image(image)
    st.markdown('<div class="frame">', unsafe_allow_html=True)
    st.image(result_image, caption="Detected Damage", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Option 2: Upload Video
st.markdown('<div class="header">Upload Video</div>', unsafe_allow_html=True)
uploaded_video = st.file_uploader("Upload a video", type=["mp4"])
if uploaded_video is not None:
    video_placeholder = st.empty()
    detect_damage_in_video(uploaded_video)

# Option 3: Live Camera
st.markdown('<div class="header">Live Camera</div>', unsafe_allow_html=True)
live_camera_button = st.button("StartLive Camera")
if live_camera_button:
    video_placeholder = st.empty()
    detect_damage_in_live_camera()
    
    
