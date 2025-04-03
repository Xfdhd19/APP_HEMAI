import streamlit as st
import PIL.Image
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO(r"weights/best.pt") # Replace with your model path

# Streamlit app
st.image(r"assets/LOGO_HEMAI.png", use_container_width=True)
st.image(r"assets/logologo.png", use_container_width=True)

st.title("Anemia Type Classifier")
st.markdown(
    """
    ⚠️**DISCLAIMER:** HEMAI only classifies 4 classes: **Healthy Blood**, **Hemolytic Anemia**,  
                    **Sickle Cell Anemia**, and **Thalassemia Anemia**.
    """
)

# Upload image
uploaded_image = st.file_uploader("📷 Please upload an image of a blood smear from a microscope view.", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Read image
    image = PIL.Image.open(uploaded_image)

    # Run inference
    results = model(image)

    # Display results
    st.image(results[0].plot(), caption="Detected Objects", use_column_width=True)
   
