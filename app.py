import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# Streamlit app title
st.title('YOLOv8 Model Deployment with Streamlit')

# Specify the correct path to your model file
model_path = '/workspaces/Practice/Practice/Practice/best1.pt'

# Load the YOLOv8 model
try:
    model = YOLO(model_path)
    model.export(format='onnx')
except Exception as e:
    st.error(f"Error loading model: {e}")

def load_image(image_file):
    img = Image.open(image_file)
    return img

def predict_back_length(image):
    results = model.predict(image, stream=True)
    back_lengths = []
    for result in results:
        img = np.array(image)  # Convert PIL image to NumPy array
        boxes = result.boxes.cpu().numpy()  # Get boxes on CPU in numpy format
        for box in boxes:  # Iterate over boxes
            r = box.xyxy[0].astype(int)  # Get corner points as int
            class_id = int(box.cls[0])  # Get class ID
            class_name = model.names[class_id]  # Get class name using the class ID
            
            if class_name == "backlength":  # Calculate hypotenuse for backlength boxes
                hypotenuse = np.linalg.norm(r[2:] - r[:2])
                back_lengths.append(hypotenuse)
                cv2.rectangle(img, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)  # Draw bounding box
                cv2.putText(img, f"Hypotenuse: {hypotenuse:.2f}", (r[0], r[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Add text
    return back_lengths, img

st.title("Elephant Back Length Measurement")
st.write("Upload an image of an elephant to measure its back length.")

image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if image_file is not None:
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    with st.spinner('Processing...'):
        back_lengths, result_img = predict_back_length(img)

    st.success('Done!')

    st.write("Detected Back Lengths (in pixels):")
    for length in back_lengths:
        st.write(f"{length:.2f} pixels")

    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption='Detected Elephants', use_column_width=True)

st.write("Note: The model detects the back length of elephants in the uploaded image.")


