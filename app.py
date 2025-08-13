# app.py
import streamlit as st
import numpy as np
import cv2
import tempfile
import pickle
from pathlib import Path

from DetectPlate import detect_plate_from_bgr
from SegmentCharacters import segment_characters

st.set_page_config(page_title="🚗 License Plate Recognition", page_icon="🚗", layout="centered")
st.title("🚗 License Plate Detection & OCR")

# ---------- Config ----------
MODEL_PATH = Path("finalized_model.sav")     # your SVM/SVC pickle
CSV_PATH   = Path("Jutraffic.csv")           # output log
DATASET_CSV = Path("dataset.csv")            # for registered check (optional)

# ---------- Load model ----------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

# ---------- Helpers ----------
def predict_plate_text(chars_20x20, x_positions):
    if not chars_20x20:
        return "", ""
    preds = []
    for ch in chars_20x20:
        feat = ch.reshape(1, -1)   # same as your training pipeline
        preds.append(model.predict(feat)[0])

    plate_raw = "".join(preds)
    # sort by x positions to get left->right ordering
    order = np.argsort(np.array(x_positions))
    plate_sorted = "".join([preds[i] for i in order])
    return plate_raw, plate_sorted

def append_csv_row(path: Path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv, datetime
    t = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    with path.open("a", newline="") as f:
        csv.writer(f).writerow([row[0], row[1], t])

def is_registered(plate: str) -> bool | None:
    if not DATASET_CSV.exists():
        return None
    try:
        content = DATASET_CSV.read_text()
        return plate in content
    except Exception:
        return None

# ---------- UI ----------
mode = st.radio("Choose input", ["Upload Image", "Upload Video"])
rotate_opt = st.selectbox("Optional rotation", [None, 90, 180, 270], index=0, format_func=lambda x: "None" if x is None else f"{x}°")

if mode == "Upload Image":
    up = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])
    if up:
        data = np.frombuffer(up.read(), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Could not read image.")
        else:
            plate_bin, box = detect_plate_from_bgr(bgr, rotate_deg=rotate_opt)
            st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), channels="RGB", caption="Input", use_column_width=True)
            if plate_bin is None:
                st.warning("No plate candidate found.")
            else:
                st.image(plate_bin*255, caption="Binary plate (inverted)", use_column_width=True)
                chars, xs = segment_characters(plate_bin)
                if not chars:
                    st.warning("No characters segmented.")
                else:
                    raw, sorted_txt = predict_plate_text(chars, xs)
                    st.success(f"Predicted (unsorted): {raw}")
                    st.success(f"Predicted (sorted): **{sorted_txt}**")

                    # CSV log
                    append_csv_row(CSV_PATH, (raw, sorted_txt))
                    reg = is_registered(sorted_txt)
                    if reg is True:
                        st.info("✅ Registered vehicle")
                    elif reg is False:
                        st.warning("❌ Not a registered vehicle")
                    else:
                        st.caption("dataset.csv not found; skipping registration check.")

else:  # Upload Video
    vf = st.file_uploader("Upload a video (mp4/mov/avi/mkv)", type=["mp4", "mov", "avi", "mkv"])
    show_preview = st.checkbox("Show preview while processing", value=True)
    max_frames = st.slider("Max frames to process (0 = all)", 0, 2000, 200, 50)

    if vf:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vf.read()); tfile.flush()
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        processed = 0
        last_prediction = None

        if not cap.isOpened():
            st.error("Could not open video.")
        else:
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    plate_bin, _ = detect_plate_from_bgr(frame, rotate_deg=rotate_opt)
                    if plate_bin is not None:
                        chars, xs = segment_characters(plate_bin)
                        if chars:
                            raw, sorted_txt = predict_plate_text(chars, xs)
                            last_prediction = (raw, sorted_txt)
                    if show_preview:
                        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
                    processed += 1
                    if max_frames and processed >= max_frames:
                        break
            finally:
                cap.release()

        st.success(f"Processed {processed} frames.")
        if last_prediction:
            raw, sorted_txt = last_prediction
            st.info(f"Last detected plate: **{sorted_txt}**")
            append_csv_row(CSV_PATH, (raw, sorted_txt))
