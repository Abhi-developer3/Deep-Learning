import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = load_model('chest_xray_cnn.h5')

st.title("Chest X-ray Pneumonia Classifier")

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["png","jpg","jpeg"])

if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(128,128))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0][0]
    
    if prediction < 0.5:
        st.success("Prediction: Normal")
    else:
        st.error("Prediction: Pneumonia")
