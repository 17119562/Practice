import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image


import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Streamlit app title
st.title('YOLOv8 Model Deployment with Streamlit')

# Specify the correct path to your model file
#model_path = 'myenv/Myproject/model/best.pt'
# Specify the correct path to your model file
model_path = 'C:/Users/lnape/Downloads/best.pt'

# Load the YOLOv8 model
model = YOLO(model_path)
# Load the model with the correct path
#model = torch.hub.load('ultralytics/yolov8', 'custom', path=model_path, force_reload=True)

def load_image(image_file):
    img = Image.open(image_file)
    return img

def predict_back_length(image):
    # Perform object detection using YOLOv8
    results = model(image)
    # Extract bounding box coordinates
    boxes = results.xyxy[0].numpy()  # assuming the first element is bounding box coordinates
    back_lengths = []
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        # Calculate back length as the distance between the points
        back_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        back_lengths.append(back_length)
    return back_lengths, results

st.title("Elephant Back Length Measurement")
st.write("Upload an image of an elephant to measure its back length.")

image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if image_file is not None:
    img = load_image(image_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    with st.spinner('Processing...'):
        back_lengths, results = predict_back_length(np.array(img))

    st.success('Done!')

    st.write("Detected Back Lengths (in pixels):")
    for length in back_lengths:
        st.write(f"{length:.2f} pixels")

    # Display the image with bounding boxes
    st.image(results.render()[0], caption='Detected Elephants', use_column_width=True)

st.write("Note: The model detects the back length of elephants in the uploaded image.")
