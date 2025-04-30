import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

st.set_page_config(page_title="Audi vs BMW Classifier", layout="centered")

IMG_SIZE = (224, 224)
MODEL_PATH = "audi_bmw_resnet_final_224.keras"

# Load model with caching so the recent results are saved
@st.cache_resource
def load_resnet_model():
    return load_model(MODEL_PATH)

model = load_resnet_model()

# Main heading/text
st.title("🚗 Audi vs BMW Image Classifier")
st.write("Upload a car image to classify it as either Audi or BMW.")

# File uploader to easily integrate images
uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    label_visibility="collapsed" 
)

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize(IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array)

    # Make prediction
    prediction = model.predict(img_preprocessed)
    confidence = prediction[0][0]

    if confidence < 0.5:
        label = "Audi"
        confidence = (1 - confidence) * 100
    else:
        label = "BMW"
        confidence = confidence * 100

    # Show prediction
    st.markdown(f"### Prediction: **{label}**")
    st.markdown(f"**Confidence:** {confidence:.2f}%")
