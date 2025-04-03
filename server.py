import streamlit as st
import PIL.Image
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO(r"weights/best.pt") # Replace with your model path

# Streamlit app
import base64

# Convert local image to Base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Get Base64 image
image_base64 = get_base64_of_image("assets/background.png")  # Ensure correct path

# Inject CSS to set the background
st.markdown(
    f"""
    <style>
    body {{
        background-image: url("data:image/png;base64,{image_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
st.image(r"assets/LOGO_HEMAI.png", use_container_width=True)
st.image(r"assets/logologo.png", use_container_width=True)

st.markdown('<h1 style="color:#B22222;">Anemia Type Classifier </h1>', unsafe_allow_html=True)
st.markdown(
    """
    <p>⚠️ <span style="color:#FF4500; font-weight: bold;">DISCLAIMER:</span>  
     HEMAI only classifies 4 classes:<span style="color: #FFD700; font-weight: bold;">Healthy Blood</span>, <span style="color: #FFD700; font-weight: bold;">Hemolytic Anemia</span>,  
    <span style="color: #FFD700; font-weight: bold;">Sickle Cell Anemia</span>, and <span style="color: #FFD700; font-weight: bold;">Thalassemia Anemia</span>.
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
    st.image(results[0].plot(), caption="Detected Objects", use_container_width=True)
   
