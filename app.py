# app.py
import io
import pickle
from pathlib import Path

import cv2
import imutils
import numpy as np
import streamlit as st
from skimage import measure
from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.measure import regionprops
from skimage.transform import resize

# ---------- Page ----------
st.set_page_config(page_title="License Plate Detector", page_icon="🚗", layout="centered")
st.title("🚗 License Plate Detection & Recognition")

# ---------- Optional OCR (EasyOCR) ----------
@st.cache_resource
def load_easyocr():
    try:
        import easyocr  # optional
        return easyocr.Reader(["en"], gpu=False)
    except Exception as e:
        return None

# ---------- Legacy SVM model (optional) ----------
@st.cache_resource(show_spinner=False)
def load_svm():
    p = Path("finalized_model.sav")
    if not p.exists():
        return None, "missing"
    try:
        with p.open("rb") as f:
            clf = pickle.load(f)
        # sanity call to ensure the attr exists on modern sklearn
        _ = getattr(clf, "predict", None)
        return clf, "ok"
    except Exception as e:
        return None, f"unreadable ({e.__class__.__name__})"

clf, clf_status = load_svm()

# ---------- Utilities ----------
def read_frame_from_video(path: Path, mode: str, frame_idx: int | None = None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    if mode == "middle":
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target = total // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    elif mode == "custom" and frame_idx is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))

    ok, frame = cap.read()
    cap.release()
    return frame if ok else None

def preprocess_for_plate(gray: np.ndarray, use_adaptive: bool, block_size: int, C: int):
    gray = cv2.equalizeHist(gray)  # contrast boost
    if use_adaptive:
        block = max(3, block_size | 1)  # must be odd
        thr = cv2.adaptiveThreshold(gray, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, block, C)
    else:
        t = threshold_otsu(gray)
        thr = (gray > t).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    return thr

def find_plate_candidates(bin_img, min_area, min_ar, max_ar):
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ar = w / float(h) if h > 0 else 0
        area = w * h
        if area < min_area:
            continue
        if not (min_ar <= ar <= max_ar):
            continue
        cnt_area = cv2.contourArea(cnt)
        if area <= 0 or cnt_area / float(area) < 0.3:  # solidity-ish
            continue
        cands.append((x, y, w, h))
    # prefer widest boxes (typical plate)
    cands.sort(key=lambda b: b[2] * b[3], reverse=True)
    return cands

def segment_characters(plate_roi_bin):
    labelled = measure.label(plate_roi_bin)
    H, W = plate_roi_bin.shape[:2]
    # heuristic character geometry
    min_h, max_h = 0.35 * H, 0.90 * H
    min_w, max_w = 0.02 * W, 0.25 * W
    chars, xs = [], []
    for r in regionprops(labelled):
        y0, x0, y1, x1 = r.bbox
        h, w = (y1 - y0), (x1 - x0)
        if h < min_h or h > max_h or w < min_w or w > max_w:
            continue
        roi = plate_roi_bin[y0:y1, x0:x1]
        # resize to 20x20 for legacy SVM
        resized = resize(roi, (20, 20), preserve_range=True).astype("float32")
        # flatten to 1D for SVM
        chars.append(resized.reshape(1, -1))
        xs.append(x0)
    # sort by x (left->right)
    if xs:
        order = np.argsort(xs)
        chars = [chars[i] for i in order]
        xs = [xs[i] for i in order]
    return chars, xs

def svm_predict(chars):
    if clf is None:
        return None
    X = np.vstack([c for c in chars]) if chars else None
    if X is None or X.size == 0:
        return ""
    try:
        preds = clf.predict(X)
        # join, mapping handled by model if needed
        return "".join(p if isinstance(p, str) else str(p) for p in preds)
    except Exception:
        return ""

def easyocr_read(reader, plate_bgr):
    if reader is None:
        return ""
    # EasyOCR expects RGB
    rgb = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB)
    try:
        res = reader.readtext(rgb, detail=0)
        # return the longest token (often the LP)
        return max(res, key=len) if res else ""
    except Exception:
        return ""

def overlay_box(img, box, color=(0, 255, 0), text=None):
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    if text:
        cv2.putText(img, text, (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# ---------- Sidebar (tuning) ----------
st.sidebar.header("Detection settings")
use_adaptive = st.sidebar.checkbox("Use adaptive threshold", value=True)
block_size = st.sidebar.slider("Adaptive block size", 11, 71, 31, step=2)
C = st.sidebar.slider("Adaptive C", 0, 20, 10, 1)
min_area = st.sidebar.slider("Min plate area (px)", 500, 10000, 2500, 100)
min_ar, max_ar = st.sidebar.slider("Aspect ratio range", 1.0, 10.0, (2.0, 6.0), 0.1)

st.sidebar.header("Recognition")
use_ocr = st.sidebar.checkbox("Use EasyOCR fallback", value=True)
reader = load_easyocr() if use_ocr else None
if use_ocr and reader is None:
    st.sidebar.warning("EasyOCR not installed. Add `easyocr` to requirements.txt to enable OCR.")

if clf is None:
    if clf_status == "missing":
        st.sidebar.info("Legacy SVM model not found (finalized_model.sav). Using OCR only if available.")
    else:
        st.sidebar.warning(f"Legacy SVM model not usable: {clf_status}. Using OCR only if available.")

st.sidebar.header("Built-in samples")
builtin_image = Path("car6.jpg")
builtin_videos = [p for p in [Path("video12.mp4"), Path("video15.mp4")] if p.exists()]
sample_choice = st.sidebar.selectbox(
    "Quick sample",
    ["—"] +
    (["car6.jpg"] if builtin_image.exists() else []) +
    [p.name for p in builtin_videos]
)

# ---------- Inputs ----------
mode = st.radio("Choose input", ["Upload image", "Upload video", "Built-in sample"])

def process_frame(frame_bgr, show_debug=False):
    view = frame_bgr.copy()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    bin_img = preprocess_for_plate(gray, use_adaptive, block_size, C)

    boxes = find_plate_candidates(bin_img, min_area, min_ar, max_ar)
    plate_text = ""
    plate_box = None

    dbg_imgs = {}

    if boxes:
        # take the best candidate first
        x, y, w, h = boxes[0]
        plate_box = (x, y, w, h)
        overlay_box(view, plate_box, (0, 255, 0))
        plate_bgr = frame_bgr[y:y+h, x:x+w].copy()

        # character segmentation on plate ROI
        plate_gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
        plate_bin = preprocess_for_plate(plate_gray, True, block_size, C)  # adaptive inside plate
        chars, xs = segment_characters(plate_bin)

        raw_svm = svm_predict(chars) or ""
        ocr_txt = easyocr_read(reader, plate_bgr) if reader is not None else ""

        # prefer OCR if non-empty, else SVM
        plate_text = ocr_txt if ocr_txt else raw_svm

        # annotate
        if plate_text:
            overlay_box(view, plate_box, (0, 200, 0), text=plate_text)

        if show_debug:
            dbg_imgs["Binary (full)"] = bin_img
            dbg_imgs["Plate ROI (BGR)"] = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB)
            dbg_imgs["Plate ROI (bin)"] = plate_bin

    return view, plate_text, plate_box, dbg_imgs

with st.expander("Training isnt properly done in Streamlit app but You can see how it works", expanded=False):
    st.markdown("""
- This app finds plate candidates via contours + geometry (area & aspect ratio), then **reads the plate** either with the **legacy SVM** (if `finalized_model.sav` loads) or **EasyOCR** (if installed).
- If detection is unstable, tune sliders in the sidebar:
  - **Adaptive threshold** block size & C
  - **Aspect ratio** and **min area**
- For videos, the app reads a **single frame** (middle by default) to avoid processing the entire file on Streamlit Cloud.
""")

# ---------- Main flow ----------
if mode == "Built-in sample" and sample_choice != "—":
    if sample_choice.endswith(".jpg"):
        frame = cv2.imread(str(builtin_image))
        if frame is None:
            st.error("Could not read built-in image.")
        else:
            out, txt, box, dbg = process_frame(frame, show_debug=True)
            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            st.success(f"Detected: **{txt or '(none)'}**")
            for name, im in dbg.items():
                st.subheader(name)
                if im.ndim == 2:
                    st.image(im, clamp=True, use_column_width=True)
                else:
                    st.image(im, channels="RGB", use_column_width=True)

    else:
        vpath = Path(sample_choice)
        frame = read_frame_from_video(vpath, mode="middle")
        if frame is None:
            st.error("Could not read a frame from the built-in video.")
        else:
            out, txt, box, dbg = process_frame(frame, show_debug=True)
            st.video(str(vpath))
            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            st.success(f"Detected: **{txt or '(none)'}**")
            for name, im in dbg.items():
                st.subheader(name)
                if im.ndim == 2:
                    st.image(im, clamp=True, use_column_width=True)
                else:
                    st.image(im, channels="RGB", use_column_width=True)

elif mode == "Upload image":
    up = st.file_uploader("Upload a JPG/PNG", type=["jpg", "jpeg", "png"])
    if up:
        data = np.frombuffer(up.read(), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Could not decode image.")
        else:
            out, txt, box, dbg = process_frame(bgr, show_debug=True)
            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            st.success(f"Detected: **{txt or '(none)'}**")
            for name, im in dbg.items():
                st.subheader(name)
                if im.ndim == 2:
                    st.image(im, clamp=True, use_column_width=True)
                else:
                    st.image(im, channels="RGB", use_column_width=True)

else:  # Upload video
    vf = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    frame_pick = st.radio("Which frame to analyze?", ["Middle frame", "First frame"], horizontal=True)
    if vf:
        # Save to temp file for OpenCV
        tmp = Path("uploaded_video.tmp")
        tmp.write_bytes(vf.read())
        mode_pick = "middle" if frame_pick == "Middle frame" else "custom"
        frame = read_frame_from_video(tmp, mode=mode_pick, frame_idx=0 if mode_pick == "custom" else None)
        if frame is None:
            st.error("Could not read a frame from the uploaded video.")
        else:
            out, txt, box, dbg = process_frame(frame, show_debug=True)
            # Show original video for context
            st.video(str(tmp))
            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            st.success(f"Detected: **{txt or '(none)'}**")
            for name, im in dbg.items():
                st.subheader(name)
                if im.ndim == 2:
                    st.image(im, clamp=True, use_column_width=True)
                else:
                    st.image(im, channels="RGB", use_column_width=True)

# ---------- Footer ----------
st.markdown(
    """
    <hr style="margin-top:2rem;margin-bottom:0.5rem;">
    <div style="text-align:center; opacity:0.7; font-size:0.9rem;">
      Built by <strong>Samiul</strong> · Questions? <a href="mailto:smhasnats@gmail.com">smhasnats@gmail.com</a>
    </div>
    """,
    unsafe_allow_html=True,
)
