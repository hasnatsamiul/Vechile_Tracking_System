# SegmentCharacters.py
import numpy as np
from skimage import measure
from skimage.measure import regionprops
from skimage.transform import resize

def segment_characters(plate_binary: np.ndarray):
    """
    Input: binary plate image (uint8: 0/1), text as white (1)
    Return:
      chars: list of (20x20) float arrays
      x_positions: list of x (column) positions for sorting
    """
    if plate_binary is None or plate_binary.size == 0:
        return [], []

    labelled_plate = measure.label(plate_binary)
    H, W = plate_binary.shape
    # Heuristic character box constraints (relative to plate size)
    min_h, max_h = 0.35*H, 0.60*H
    min_w, max_w = 0.05*W, 0.15*W

    chars, x_positions = [], []
    for regions in regionprops(labelled_plate):
        y0, x0, y1, x1 = regions.bbox
        h, w = (y1 - y0), (x1 - x0)
        if (min_h <= h <= max_h) and (min_w <= w <= max_w):
            roi = plate_binary[y0:y1, x0:x1].astype(np.float32)
            resized = resize(roi, (20, 20), preserve_range=True, anti_aliasing=True).astype(np.float32)
            chars.append(resized)
            x_positions.append(x0)

    return chars, x_positions
