import streamlit as st
import PIL.Image
import os
import requests
from ultralytics import YOLO

model_url = "https://drive.google.com/drive/u/2/folders/1-SaaH2TFedzV5cVBSAdNoH4dA1N6FWrd?fbclid=IwZXh0bgNhZW0CMTEAAR19kKDegJDvjxrQfAW6LY5bBInAs_s5ejKh1FkZ-t4sW4ptdbiZuRYtyM0_aem_EAKmyqNFE05CbAFhU9rOAA"  # Replace with actual link
model_path = "./best.pt"

# Ensure directory exists
os.makedirs("weights", exist_ok=True)

# Download model if not present
if not os.path.exists(model_path):
    print("Downloading model...")
    response = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)
    print("Download complete!")

model = YOLO(model_path)


# Streamlit app
st.image(r"assets/LOGO_HEMAI.png", use_column_width=True)

st.title("Anemia Type Classifier")
st.markdown(
    """
    **DISCLAIMER:** HEMAI only classifies 4 classes: **Healthy Blood**, **Hemolytic Anemia**,  
                    **Sickle Cell Anemia**, and **Thalassemia Anemia**.
    """
)

# Upload image
uploaded_image = st.file_uploader("Please upload an image of a bloodsmear", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Read image
    image = PIL.Image.open(uploaded_image)

    # Run inference
    results = model(image)

    # Display results
    st.image(results[0].plot(), caption="Detected Objects", use_column_width=True)
