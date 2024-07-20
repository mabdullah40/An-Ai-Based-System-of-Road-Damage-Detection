# An-Ai-Based-System-of-Road-Damage-Detection

This project is a road damage detection application built using Streamlit and YOLO v8. The application allows users to upload photos, videos, or use a live camera feed to detect various types of road damages in real-time.

## Features

- **Image, Video, and Live Camera Feed**: Upload images or videos, or use a live camera feed to detect road damages.
- **Real-Time Detection**: Real-time damage detection using the YOLO v8 model.
- **Damage Types**: Detects seven types of road damages:
  - Alligator Cracks
  - Damaged Crosswalk
  - Damaged Paint
  - Longitudinal Cracks
  - Manhole Cover
  - Potholes
  - Transverse Cracks
- **PDF Report Generation**: Generate a PDF report summarizing the detected road damages, including the total number of damages and the count of each type of damage.

## Dataset 
This dataset is available on kaggle

## Add Efficiency
- Due to the shortage of resourse this model is trained for 20 Epochs you can increse it epoch to make the model much more efficent.
- Must have an eye on the model perfformance to not to overfit it.

## Technical Details

- **Framework**: Streamlit for building the web application.
- **Object Detection**: YOLO v8 for detecting and classifying road damages.
- **Image and Video Processing**: OpenCV (`cv2`) for handling image and video input.
- **Image Manipulation**: Pillow (`PIL`) for image processing tasks.
- **Data Handling**: NumPy for numerical operations on image data.
- **Visualization**: Matplotlib for plotting and visualizing data.
- **PDF Generation**: FPDF for creating and exporting PDF reports.
- **Auxiliary Libraries**:
  - **cvzone**: For additional computer vision utilities.
  - **sort**: For object tracking using the SORT algorithm.
  - **tempfile** and **os**: For handling temporary files and file operations.
  - **time** and **datetime**: For managing time and date operations.

### YOLO v8 Model

- **Model Type**: YOLO v8 (You Only Look Once version 8)
- **Usage**: Real-time object detection and classification
- **Detection Types**: Detects various types of road damages based on trained model weights.

### Dependencies

The project uses the following Python libraries, as specified in `requirements.txt`:

- `streamlit`
- `opencv-python-headless`
- `Pillow`
- `numpy`
- `ultralytics`
- `cvzone`
- `sort`
- `matplotlib`
- `fpdf`
