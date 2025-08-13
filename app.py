# app.py
import streamlit as st
import numpy as np
import cv2
from pathlib import Path
import tempfile
import pickle

from DetectPlate import detect_plate_from_bgr
from SegmentCharacters import segment_characters

st.set_page_config(page_title="License Plate Detector", page_icon="🚗", layout="centered")
st.title("🚗 License Plate Detection (Streamlit)")

st.sidebar.header("Settings")
rotate_deg = st.sidebar.selectbox("Rotate image (deg)", [None, 90, 180, 270], index=0)

mode = st.radio("Choose input", ["Upload Image", "Upload Video"])

# Optional: load your classifier (if you have finalized_model.sav)
@st.cache_resource
def load_clf():
    p = Path("finalized_model.sav")
    if p.exists():
        with p.open("rb") as f:
            return pickle.load(f)
    return None

clf = load_clf()

def chars_to_string(chars, x_positions):
    if not chars:
        return ""
    order = np.argsort(np.array(x_positions))
    # If you have a classifier, predict each 20x20 character here
    if clf is not None:
        preds = []
        for i in order:
            ch = chars[i].reshape(1, -1)  # expected shape for classic ML
            pred = clf.predict(ch)[0]
            preds.append(pred)
        return "".join(preds)
    else:
        # Without a classifier, just count segments
        return f"[{len(order)} chars detected]"

if mode == "Upload Image":
    up = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])
    if up:
        data = np.frombuffer(up.read(), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Could not read image.")
        else:
            plate_bin, box = detect_plate_from_bgr(bgr, rotate_deg)
            st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Input image", use_container_width=True)
            if plate_bin is None:
                st.warning("No plate candidate found.")
            else:
                chars, xs = segment_characters(plate_bin)
                plate_text = chars_to_string(chars, xs)
                st.image(plate_bin * 255, caption=f"Plate (binary). Prediction: {plate_text}", use_container_width=True)

else:
    vf = st.file_uploader("Upload a short video", type=["mp4", "mov", "avi", "mkv"])
    max_frames = st.sidebar.slider("Max frames (0 = all)", 0, 1000, 200, 50)
    show_preview = st.checkbox("Show preview frames", value=True)

    if vf:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vf.read()); tfile.flush()
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        frames_done = 0
        all_predictions = []

        if not cap.isOpened():
            st.error("Could not open video.")
        else:
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    plate_bin, box = detect_plate_from_bgr(frame, rotate_deg)
                    if plate_bin is not None:
                        chars, xs = segment_characters(plate_bin)
                        plate_text = chars_to_string(chars, xs)
                        all_predictions.append(plate_text)
                    if show_preview:
                        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    frames_done += 1
                    if max_frames and frames_done >= max_frames:
                        break
            finally:
                cap.release()

        st.success(f"Processed {frames_done} frames.")
        if all_predictions:
            st.write("Detections (sample):", all_predictions[:10])
