import cv2
import numpy as np

def detect_skin(filename: str):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(filename)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))
    mask_hsv = cv2.inRange(hsv, np.array([0, 40, 50], dtype=np.uint8), np.array([25, 255, 255], dtype=np.uint8))
    combined = cv2.bitwise_and(mask_ycrcb, mask_hsv)
    combined = cv2.medianBlur(combined, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=3)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    combined = cv2.dilate(combined, kernel, iterations=1)
    out = np.zeros_like(img)
    out[combined == 255] = [255, 255, 255]
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)