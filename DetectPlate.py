# DetectPlate.py
import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.measure import regionprops

def _find_plate_regions(binary_car_image, plate_dimensions):
    label_image = measure.label(binary_car_image)
    min_h, max_h, min_w, max_w = plate_dimensions
    plate_rois = []
    boxes = []
    for region in regionprops(label_image):
        if region.area < 50:
            continue
        min_row, min_col, max_row, max_col = region.bbox
        h = max_row - min_row
        w = max_col - min_col
        if (min_h <= h <= max_h) and (min_w <= w <= max_w) and (w > h):
            roi = binary_car_image[min_row:max_row, min_col:max_col]
            plate_rois.append(roi)
            boxes.append((min_row, min_col, max_row, max_col))
    return plate_rois, boxes

def detect_plate_from_bgr(bgr_img: np.ndarray, rotate_deg: int | None = None):
    """
    Input: BGR image
    Return:
      plate_binary (np.ndarray | None),
      box (tuple | None)  -> (min_row, min_col, max_row, max_col) in grayscale space
    """
    if bgr_img is None or bgr_img.size == 0:
        return None, None

    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
    if rotate_deg is not None:
        # rotate around center, keeping size
        (h, w) = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), rotate_deg, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    gray_255 = (gray * 255.0)
    thr = threshold_otsu(gray_255)
    binary = (gray_255 > thr).astype(np.uint8)

    H, W = binary.shape
    dims1 = (0.03*H, 0.08*H, 0.15*W, 0.30*W)
    dims2 = (0.08*H, 0.20*H, 0.15*W, 0.40*W)

    rois, boxes = _find_plate_regions(binary, dims1)
    if not rois:
        rois, boxes = _find_plate_regions(binary, dims2)

    if not rois:
        return None, None

    # choose the largest candidate
    areas = [r.shape[0]*r.shape[1] for r in rois]
    idx = int(np.argmax(areas))
    plate_bin = rois[idx]
    box = boxes[idx]
    # invert so text becomes white on black (as your old code assumed)
    plate_bin = (1 - plate_bin).astype(np.uint8)
    return plate_bin, box
