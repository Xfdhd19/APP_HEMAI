import streamlit as st
import PIL.Image
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO(r"weights/best.pt") # Replace with your model path

# Streamlit app
st.image(r"assets/LOGO_HEMAI.png", use_container_width=True)
st.image(r"assets/logologo.png", use_container_width=True)

st.markdown('<h1 style="color:#FF5733;">Anemia Type Classifier </h1>', unsafe_allow_html=True)
st.markdown(
    """
    <p>⚠️ <span style="color: orange; font-weight: bold;">DISCLAIMER:</span>  
    HEMAI only classifies 4 classes:  
    <strong>Healthy Blood</strong>, <strong>Hemolytic Anemia</strong>,  
    <strong>Sickle Cell Anemia</strong>, and <strong>Thalassemia Anemia</strong>.
    </p>
    """, unsafe_allow_html=True
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
   
