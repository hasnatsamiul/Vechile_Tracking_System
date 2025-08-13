# app.py
import streamlit as st
import cv2
import numpy as np
from SegmentCharacters import segment_characters  # whatever you expose
from DetectPlate import detect_plate              # whatever you expose
import joblib

st.set_page_config(page_title="License Plate Detector", page_icon="🚗")
st.title("🚗 License Plate Detection & OCR")

# load your trained model
@st.cache_resource
def load_model():
    return joblib.load("finalized_model.sav")

model = load_model()

uploaded = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])
if uploaded:
    data = np.frombuffer(uploaded.read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Could not read image.")
    else:
        plate_img = detect_plate(bgr)                      # returns cropped plate
        chars = segment_characters(plate_img)              # list of glyph images
        # predict each char with your SVC
        preds = []
        for ch in chars:
            # reshape/featureize exactly as in your training code:
            feat = ch.reshape(1, -1)                       # example; match your TrainRecognizeCharacters.py
            preds.append(model.predict(feat)[0])

        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Input", channels="RGB")
        if plate_img is not None:
            st.image(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB), caption="Detected Plate", channels="RGB")
        st.success(f"Predicted plate: {''.join(preds)}")
