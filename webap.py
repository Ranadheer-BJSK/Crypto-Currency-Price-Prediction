import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# Load the trained model
MODEL_PATH = "brain_stroke_model.h5"
model = load_model(MODEL_PATH)

# Define class labels
class_labels = {0: "Hemorrhagic", 1: "Ischemic"}

# Excel file to store patient data
EXCEL_FILE = "patient_data.xlsx"

def preprocess_image(image):
    image = image.resize((128, 128))  # Resize to model's input shape
    image = img_to_array(image) / 255.0  # Normalize pixel values
    if image.shape[-1] == 1:  # Convert grayscale to RGB
        image = np.stack((image,) * 3, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

def predict_label(image):
    image = preprocess_image(image)
    prediction = model.predict(image)
    print("Raw Model Prediction:", prediction)  # Debugging line
    
    if model.output_shape[-1] == 1:  # Sigmoid output (binary classification)
        label = int(prediction[0][0] > 0.5)
        confidence = float(prediction[0][0]) if label == 1 else float(1 - prediction[0][0])
    else:
        label = np.argmax(prediction)  # Softmax output (multi-class classification)
        confidence = float(np.max(prediction))
    
    return class_labels[label], confidence

def save_to_excel(patient_data):
    if os.path.exists(EXCEL_FILE):
        df = pd.read_excel(EXCEL_FILE)
    else:
        df = pd.DataFrame(columns=["Patient ID", "Name", "Age", "Gender", "Prediction", "Confidence"])
    df = pd.concat([df, pd.DataFrame([patient_data])], ignore_index=True)
    df.to_excel(EXCEL_FILE, index=False)

# Streamlit UI
st.title("Brain Stroke Prediction System")
st.header("Enter Patient Details")

patient_id = st.text_input("Patient ID")
name = st.text_input("Patient Name")
age = st.number_input("Age", min_value=0, max_value=120, step=1)
gender = st.radio("Gender", ("Male", "Female", "Other"))
image_file = st.file_uploader("Upload Brain CT Scan", type=["jpg", "png", "jpeg"])

if st.button("Predict"):
    if patient_id and name and image_file:
        image = Image.open(image_file).convert("RGB")  # Ensure image is in RGB mode
        prediction, confidence = predict_label(image)
        
        st.write(f"Prediction: **{prediction}**")
        st.write(f"Confidence: **{confidence:.2f}**")
        
        # Save data
        patient_data = {
            "Patient ID": patient_id,
            "Name": name,
            "Age": age,
            "Gender": gender,
            "Prediction": prediction,
            "Confidence": confidence
        }
        save_to_excel(patient_data)
        st.success("Patient data saved successfully!")
    else:
        st.error("Please fill all fields and upload an image.")
