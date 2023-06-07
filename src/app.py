import streamlit as st
import torch
from PIL import Image
import cv2
from functions import *

coco_classes = ['person',  'bicycle',  'car',  'motorcycle',  'airplane',  'bus',  'train',  'truck',  'boat',  'traffic light',  'fire hydrant',  'stop sign',  'parking meter',  'bench',  'bird',  'cat',  'dog',  'horse',  'sheep',  'cow',  'elephant',  'bear',  'zebra',  'giraffe',  'backpack',  'umbrella',  'handbag',  'tie',  'suitcase',  'frisbee',  'skis',  'snowboard',  'sports ball',  'kite',  'baseball bat',  'baseball glove',  'skateboard',  'surfboard',  'tennis racket',  'bottle',  'wine glass',  'cup',  'fork',  'knife',  'spoon',  'bowl',  'banana',  'apple',  'sandwich',  'orange',  'broccoli',  'carrot',  'hot dog',  'pizza',  'donut',  'cake',  'chair',  'couch',  'potted plant',  'bed',  'dining table',  'toilet',  'tv',  'laptop',  'mouse',  'remote',  'keyboard',  'cell phone',  'microwave',  'oven',  'toaster',  'sink',  'refrigerator',  'book',  'clock',  'vase',  'scissors',  'teddy bear',  'hair drier',  'toothbrush']

# Main function to run the Streamlit app
def main():
    # Logo
    img = Image.open('logo.png')
    # Display the image as a logo at the top of the page
    st.image(img, use_column_width=True)
    # Set the title of the app
    st.title("YOLOv8 Model Test on COCO")
    # Create a dropdown select box to choose the model
    model_type = st.selectbox('Select your model', ['Object Detection', 'Pose Estimation', 'Segmentation'])

    selected_model_version = select_model_version(model_type)
    
    selected_classes = select_classes(model_type)
    
    selected_confidence = st.slider('Min Confidence Score Threshold', min_value = 0.0, max_value = 1.0, value = 0.4)

    uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4"])

    # Check if a file is uploaded
    if uploaded_file is not None:

        # Show the details of the uploaded file
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(file_details)

        
        # If an image file is uploaded
        if uploaded_file.type.startswith('image/'):
            # Load the image
            orig_image = load_image(uploaded_file)
            
            # Display the uploaded image
            st.image(orig_image, caption='Uploaded Image.', use_column_width=True)
            
            # Create a button to start the inference
            if st.button('Infer'):
                # Load the selected model
                model = load_model(selected_model_version)

                with st.spinner('Detecting...'):
                    run_folder = create_run_folder('images')
                    save_image(orig_image, f'{run_folder}/{uploaded_file.name}')
                    image = inference(orig_image, model, selected_classes, selected_confidence)
                    save_image(image, f'{run_folder}/output.jpg')
                    st.image(image, caption='Detected Objects.', use_column_width=True)

        # If a video file is uploaded
        elif uploaded_file.type.startswith('video/'):
            tracking = st.checkbox('Add traking')
            
            if st.button('Infer video'):
                run_folder = create_run_folder('videos')
                # Load the selected model
                model = load_model(selected_model_version)

                input_path =f'{run_folder}/{uploaded_file.name}'
                output_path = f'{run_folder}/output.mp4'
                with open(input_path, 'wb') as out_file:
                    out_file.write(uploaded_file.getbuffer())
                with st.spinner('Processing video...'):
                    convert_video(input_path)
                st.write("Original:")
                st.video(input_path)
                with st.spinner('Detecting...'):
                    process_video(input_path, output_path, model, selected_classes, selected_confidence, tracking)
                    st.write("Output:")
                    st.video(output_path)

if __name__ == "__main__":
   main()
