import streamlit as st
import torch
from PIL import Image
import cv2
import numpy as np
import subprocess
import shutil
import datetime
from pathlib import Path
from ultralytics import YOLO


coco_classes = ['person',  'bicycle',  'car',  'motorcycle',  'airplane',  'bus',  'train',  'truck',  'boat',  'traffic light',  'fire hydrant',  'stop sign',  'parking meter',  'bench',  'bird',  'cat',  'dog',  'horse',  'sheep',  'cow',  'elephant',  'bear',  'zebra',  'giraffe',  'backpack',  'umbrella',  'handbag',  'tie',  'suitcase',  'frisbee',  'skis',  'snowboard',  'sports ball',  'kite',  'baseball bat',  'baseball glove',  'skateboard',  'surfboard',  'tennis racket',  'bottle',  'wine glass',  'cup',  'fork',  'knife',  'spoon',  'bowl',  'banana',  'apple',  'sandwich',  'orange',  'broccoli',  'carrot',  'hot dog',  'pizza',  'donut',  'cake',  'chair',  'couch',  'potted plant',  'bed',  'dining table',  'toilet',  'tv',  'laptop',  'mouse',  'remote',  'keyboard',  'cell phone',  'microwave',  'oven',  'toaster',  'sink',  'refrigerator',  'book',  'clock',  'vase',  'scissors',  'teddy bear',  'hair drier',  'toothbrush']


    

# Function to load the model, using @st.cache_resource for optimized loading

def load_model(model_version):
    return YOLO(model_version)

# Function to save an image to a specified path
def save_image(image, path):
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# Function to load an image from a file
def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img)

# Function to perform prediction on an image using a specified model
def predict(image, model):
    img = np.array(image)
    results = model(img)
    return results

# Function to convert a video file to the MP4 format using FFMPEG
def convert_video(path):
    temp_path = path + '.temp.mp4'
    subprocess.call(['ffmpeg', '-y', '-i', path, '-c:v', 'libx264', temp_path])
    shutil.move(temp_path, path)

# Function to perform inference on a frame using a specified model and selected classes
def inference(frame, model, wanted_classes, selected_confidence):
    res = model.predict(source=frame, conf=selected_confidence, classes=wanted_classes)
    res_plotted = res[0].plot()
    return res_plotted
    
def select_model_version(model_type):
    if model_type == 'Object Detection':
        selected_version = st.selectbox('Select your model version', ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l'])
    elif model_type == 'Pose Estimation':
        selected_version = st.selectbox('Select your model version', ['yolov8n-pose', 'yolov8s-pose', 'yolov8m-pose', 'yolov8l-pose'])
    elif model_type == 'Segmentation':
        selected_version = st.selectbox('Select your model version', ['yolov8n-seg', 'yolov8s-seg', 'yolov8m-seg', 'yolov8l-seg'])
    return "models/"+selected_version

# Function to select classes from the model
def select_classes(model_type):
    if model_type == 'Pose Estimation':
        selected_classes = [0]
    else:
        selected_classes = st.multiselect("Select Classes", range(len(coco_classes)), format_func = lambda x: coco_classes[x])
        # When no classes are selected, select all classes
        if len(selected_classes) == 0:
            selected_classes = range(len(coco_classes))    
    return selected_classes

# Function to process a video
def process_video(input_path, output_path, model, wanted_classes, selected_confidence, tracking = False):
    # Load the video
    cap = cv2.VideoCapture(input_path)
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create a progress bar
    progress_bar = st.progress(0)
    
    # Initialize the counter for processed frames
    processed_frames = 0

    # Loop through all frames in the video
    while cap.isOpened():
        print(f'Processing frame: {processed_frames}/{total_frames}')
        ret, frame = cap.read()
        if not ret:
            break
        if tracking:
            frame_bgr = model.track(frame, persist=True,classes=wanted_classes)[0].plot()
        else:
            frame_bgr = inference(frame, model, wanted_classes, selected_confidence)
        out.write(frame_bgr)

        # Update the progress bar
        processed_frames += 1
        progress_bar.progress(processed_frames / total_frames)

    # Release video resources when done
    cap.release()
    out.release()
    with st.spinner('Converting:'):
        convert_video(output_path)


# Function to create a new directory for each run
def create_run_folder(folder_name):
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = f'{folder_name}/run_{timestamp}'
    Path(run_folder).mkdir(parents=True, exist_ok=True)
    return run_folder
