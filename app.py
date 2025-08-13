import sys, types, io, tempfile, csv, pickle, time
from pathlib import Path
from datetime import datetime

import streamlit as st
import numpy as np
import cv2

# ---- scikit-image pieces (compact imports to speed cold start)
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.measure import regionprops
from skimage.transform import resize

# =========================
# Compat shim for old pickles
# =========================
# Older scikit-learn pickles reference `sklearn.svm.classes`.
# On modern versions it's `sklearn.svm._classes`. This shim maps the legacy path.
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
    # If sklearn isn't importable yet, we'll fail later in load_clf() anyway.
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
    p = Path("finalized_model.sav")
    if not p.exists():
        return None
    with p.open("rb") as f:
        return pickle.load(f)

clf = load_clf()
if clf is None:
    st.warning("⚠️ `finalized_model.sav` not found at repo root. Upload it to enable recognition.")

# =========================
# Core CV helpers
# =========================
def to_gray(image_bgr: np.ndarray) -> np.ndarray:
    """BGR (OpenCV) -> single-channel float grayscale in [0,255]."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)

def maybe_rotate(image: np.ndarray, deg: int) -> np.ndarray:
    if deg == 0:
        return image
    # Use cv2 rotation for speed
    if deg == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

def binarize(gray_255: np.ndarray) -> np.ndarray:
    """Otsu threshold; return boolean mask (True = foreground)."""
    t = threshold_otsu(gray_255)
    return gray_255 > t

def find_plate(binary: np.ndarray, min_area: int) -> tuple[np.ndarray | None, tuple[int,int,int,int] | None]:
    """
    Heuristic plate finder using connected components and aspect/size filters.
    Returns (plate_binary_crop, bbox) or (None, None).
    """
    label_image = measure.label(binary)
    H, W = label_image.shape

    # Two plausible dimension sets (from your script):
    dims1 = (0.03*H, 0.08*H, 0.15*W, 0.30*W)
    dims2 = (0.08*H, 0.20*H, 0.15*W, 0.40*W)

    def search(dims):
        min_h, max_h, min_w, max_w = dims
        for region in regionprops(label_image):
            if region.area < min_area:
                continue
            y0, x0, y1, x1 = region.bbox
            rh, rw = (y1 - y0), (x1 - x0)
            # typical plate: width > height, fits size window
            if (min_h <= rh <= max_h) and (min_w <= rw <= max_w) and (rw > rh):
                crop = binary[y0:y1, x0:x1]
                return crop, (y0, x0, y1, x1)
        return None, None

    plate, bbox = search(dims1)
    if plate is None:
        plate, bbox = search(dims2)
    return plate, bbox

def segment_characters(license_plate_bool: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
    """
    Invert license plate, label, then pick character-like regions by size.
    Returns (list of 20x20 float arrays), (x positions for ordering).
    """
    license_inv = np.invert(license_plate_bool)  # match original approach
    labelled = measure.label(license_inv)

    H, W = license_inv.shape
    # Heuristic from your script:
    char_dims = (0.35*H, 0.60*H, 0.05*W, 0.15*W)
    min_h, max_h, min_w, max_w = char_dims

    chars = []
    x_positions = []

    for region in regionprops(labelled):
        y0, x0, y1, x1 = region.bbox
        rh, rw = (y1 - y0), (x1 - x0)
        if (min_h < rh < max_h) and (min_w < rw < max_w):
            roi = license_inv[y0:y1, x0:x1]
            # resize to 20x20 float32 in [0,1]
            r = resize(roi.astype(float), (20, 20), preserve_range=True, anti_aliasing=True)
            r = (r > 0.5).astype(np.float32)  # simple binarization after resize
            chars.append(r)
            x_positions.append(int(x0))
    return chars, x_positions

def predict_plate(chars: list[np.ndarray], x_positions: list[int]):
    """Predict with classifier; return raw and ordered strings."""
    if not chars or clf is None:
        return "", ""
    # Prepare features: flatten to 1D
    X = np.array([c.reshape(1, -1)[0] for c in chars], dtype=np.float32)
    preds = clf.predict(X)  # each item likely a single char label

    raw = "".join(str(p) for p in preds)

    # order by x (left->right)
    order = np.argsort(np.array(x_positions))
    ordered = "".join(str(preds[i]) for i in order)
    return raw, ordered

def draw_bbox(img_bgr: np.ndarray, bbox: tuple[int,int,int,int] | None) -> np.ndarray:
    if bbox is None:
        return img_bgr
    y0, x0, y1, x1 = bbox
    out = img_bgr.copy()
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2)
    return out

# =========================
# UI – choose input
# =========================
mode = st.radio("Choose input", ["Image", "Video (first frame)"])

uploaded = None
frame_bgr = None

if mode == "Image":
    uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
    if uploaded:
        data = np.frombuffer(uploaded.read(), np.uint8)
        frame_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

else:
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

if frame_bgr is not None:
    # Optional rotate
    frame_bgr = maybe_rotate(frame_bgr, rotate_deg)

    # Show original
    st.subheader("Input")
    st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    # Gray + binary
    gray = to_gray(frame_bgr)
    binary = binarize(gray)

    # Find plate region
    plate_bool, bbox = find_plate(binary, min_region_area)

    # Visualize bbox on original
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

    # Segment & predict
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

            # Log to CSV like your script
            now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            row = [raw, ordered, now]
            try:
                with open("Jutraffic.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
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
