import sys, types, io, tempfile, csv, pickle, time
from pathlib import Path
from datetime import datetime

import streamlit as st
import numpy as np
import cv2

from skimage.filters import threshold_otsu
from skimage import measure
from skimage.measure import regionprops
from skimage.transform import resize

# ======== compat shim for old sklearn pickles (sklearn.svm.classes) ========
try:
    from sklearn.svm import SVC, LinearSVC, SVR, LinearSVR, NuSVC, NuSVR, OneClassSVM
    legacy = types.ModuleType("sklearn.svm.classes")
    legacy.SVC = SVC
    legacy.LinearSVC = LinearSVC
    legacy.SVR = SVR
    legacy.LinearSVR = LinearSVR
    legacy.NuSVC = NuSVC
    legacy.NuSVR = NuSVR
    legacy.OneClassSVM = OneClassSVM
    sys.modules["sklearn.svm.classes"] = legacy
except Exception:
    pass

# =========================
# Streamlit page setup
# =========================
st.set_page_config(page_title="License Plate Detector", page_icon="🚗", layout="centered")
st.title("🚗 License Plate Detection & Recognition")

st.sidebar.header("Settings")
rotate_deg = st.sidebar.selectbox("Rotate image (if needed)", [0, 90, 180, 270], index=0)
min_region_area = st.sidebar.slider("Min region area (noise filter)", 10, 300, 50, 10)
show_intermediates = st.sidebar.checkbox("Show intermediate images", value=False)

# =========================
# Model loader
# =========================
@st.cache_resource(show_spinner=True)
def load_clf():
    import pickle, types, sys
    from pathlib import Path

    # ---- compat shim for very old pickles that reference sklearn.svm.classes
    try:
        from sklearn.svm import SVC, LinearSVC, SVR, LinearSVR, NuSVC, NuSVR, OneClassSVM
        legacy = types.ModuleType("sklearn.svm.classes")
        legacy.SVC = SVC
        legacy.LinearSVC = LinearSVC
        legacy.SVR = SVR
        legacy.LinearSVR = LinearSVR
        legacy.NuSVC = NuSVC
        legacy.NuSVR = NuSVR
        legacy.OneClassSVM = OneClassSVM
        sys.modules["sklearn.svm.classes"] = legacy
    except Exception:
        pass
    # ---- end compat shim

    p = Path("finalized_model.sav")
    if not p.exists():
        return None
    with p.open("rb") as f:
        clf = pickle.load(f)

    # ---- patch missing attrs from older sklearn versions
    try:
        # Only for SVM-like models
        name = getattr(clf, "__class__", type(clf)).__name__
        if name in ("SVC", "LinearSVC", "NuSVC"):
            if not hasattr(clf, "break_ties"):
                clf.break_ties = False  # default in modern sklearn
            if not hasattr(clf, "decision_function_shape"):
                clf.decision_function_shape = "ovr"  # safe default
    except Exception:
        pass
    # ---- end patch

    return clf


# =========================
# Core CV helpers
# =========================
def to_gray(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)

def maybe_rotate(image: np.ndarray, deg: int) -> np.ndarray:
    if deg == 0:
        return image
    if deg == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

def binarize(gray_255: np.ndarray) -> np.ndarray:
    t = threshold_otsu(gray_255)
    return gray_255 > t

def find_plate(binary: np.ndarray, min_area: int):
    label_image = measure.label(binary)
    H, W = label_image.shape

    dims1 = (0.03*H, 0.08*H, 0.15*W, 0.30*W)
    dims2 = (0.08*H, 0.20*H, 0.15*W, 0.40*W)

    def search(dims):
        min_h, max_h, min_w, max_w = dims
        for region in regionprops(label_image):
            if region.area < min_area:
                continue
            y0, x0, y1, x1 = region.bbox
            rh, rw = (y1 - y0), (x1 - x0)
            if (min_h <= rh <= max_h) and (min_w <= rw <= max_w) and (rw > rh):
                crop = binary[y0:y1, x0:x1]
                return crop, (y0, x0, y1, x1)
        return None, None

    plate, bbox = search(dims1)
    if plate is None:
        plate, bbox = search(dims2)
    return plate, bbox

def segment_characters(license_plate_bool: np.ndarray):
    license_inv = np.invert(license_plate_bool)
    labelled = measure.label(license_inv)

    H, W = license_inv.shape
    char_dims = (0.35*H, 0.60*H, 0.05*W, 0.15*W)
    min_h, max_h, min_w, max_w = char_dims

    chars, x_positions = [], []
    for region in regionprops(labelled):
        y0, x0, y1, x1 = region.bbox
        rh, rw = (y1 - y0), (x1 - x0)
        if (min_h < rh < max_h) and (min_w < rw < max_w):
            roi = license_inv[y0:y1, x0:x1]
            r = resize(roi.astype(float), (20, 20), preserve_range=True, anti_aliasing=True)
            r = (r > 0.5).astype(np.float32)
            chars.append(r)
            x_positions.append(int(x0))
    return chars, x_positions

def predict_plate(chars, x_positions):
    if not chars or clf is None:
        return "", ""
    X = np.array([c.reshape(1, -1)[0] for c in chars], dtype=np.float32)
    preds = clf.predict(X)
    raw = "".join(str(p) for p in preds)
    order = np.argsort(np.array(x_positions))
    ordered = "".join(str(preds[i]) for i in order)
    return raw, ordered

def draw_bbox(img_bgr: np.ndarray, bbox):
    if bbox is None:
        return img_bgr
    y0, x0, y1, x1 = bbox
    out = img_bgr.copy()
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2)
    return out

# =========================
# Sample files (built-in)
# =========================
SAMPLE_IMAGE = Path("car6.jpg")
SAMPLE_VIDEOS = [p for p in [Path("video12.mp4"), Path("video15.mp4")] if p.exists()]

has_sample_image = SAMPLE_IMAGE.exists()
has_sample_videos = len(SAMPLE_VIDEOS) > 0

# =========================
# UI – choose source
# =========================
source = st.radio(
    "Choose input source",
    [
        "Sample image" if has_sample_image else "Upload image",
        "Upload image",
        "Sample video (first frame)" if has_sample_videos else "Upload video",
        "Upload video",
    ],
    index=0,
)

frame_bgr = None

# ---- Sample image
if source == "Sample image" and has_sample_image:
    st.caption(f"Using bundled sample: `{SAMPLE_IMAGE.name}`")
    data = SAMPLE_IMAGE.read_bytes()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    frame_bgr = img

# ---- Upload image
elif source == "Upload image":
    uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
    if uploaded:
        data = np.frombuffer(uploaded.read(), np.uint8)
        frame_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

# ---- Sample video
elif source == "Sample video (first frame)" and has_sample_videos:
    vid_name = st.selectbox("Choose bundled sample video", [p.name for p in SAMPLE_VIDEOS], index=0)
    chosen = next(p for p in SAMPLE_VIDEOS if p.name == vid_name)
    st.caption(f"Using bundled video: `{chosen.name}` (processing first frame)")
    cap = cv2.VideoCapture(str(chosen))
    ok, f = cap.read()
    cap.release()
    if ok:
        frame_bgr = f
    else:
        st.error("Could not read a frame from the sample video.")

# ---- Upload video
elif source == "Upload video":
    vfile = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if vfile:
        t = tempfile.NamedTemporaryFile(delete=False)
        t.write(vfile.read())
        t.flush()
        cap = cv2.VideoCapture(t.name)
        ok, f = cap.read()
        cap.release()
        if ok:
            frame_bgr = f
        else:
            st.error("Could not read a frame from the uploaded video.")

# =========================
# Run pipeline
# =========================
if frame_bgr is not None:
    frame_bgr = maybe_rotate(frame_bgr, rotate_deg)

    st.subheader("Input")
    st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    gray = to_gray(frame_bgr)
    binary = binarize(gray)

    plate_bool, bbox = find_plate(binary, min_region_area)

    vis = draw_bbox(frame_bgr, bbox)
    st.subheader("Detection")
    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    if show_intermediates:
        st.caption("Intermediate views")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(gray.astype(np.uint8), clamp=True, use_column_width=True, caption="Grayscale")
        with col2:
            st.image((binary * 255).astype(np.uint8), clamp=True, use_column_width=True, caption="Binary (Otsu)")
        with col3:
            if plate_bool is not None:
                st.image((plate_bool * 255).astype(np.uint8), clamp=True, use_column_width=True, caption="Plate crop (binary)")

    if plate_bool is None:
        st.error("No plausible plate region found. Try rotating, different image, or tweak settings.")
    else:
        chars, xs = segment_characters(plate_bool)
        st.write(f"Characters detected: **{len(chars)}**")

        raw, ordered = predict_plate(chars, xs)
        if not ordered:
            st.warning("Model not loaded or no characters segmented — cannot predict.")
        else:
            st.success(f"**Predicted plate (ordered):** `{ordered}`")
            st.caption(f"Raw (detection order): `{raw}`")

            now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            row = [raw, ordered, now]
            try:
                with open("Jutraffic.csv", "a", newline="") as f:
                    csv.writer(f).writerow(row)
                st.info("Logged to Jutraffic.csv")
            except Exception as e:
                st.warning(f"Could not write to Jutraffic.csv: {e}")

    st.write("---")

# =========================
# Footer
# =========================
st.markdown(
    """
    <div style="text-align:center; font-size:0.95rem; opacity:0.75; margin-top:2rem;">
      Built by <strong>Samiul</strong>. For issues: <a href="mailto:smhasnats@gmail.com">smhasnats@gmail.com</a>
    </div>
    """,
    unsafe_allow_html=True,
)
