"""modules/thermal/thermal_model.py — Hotspot detection on thermal images."""
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io, cv2
import numpy as np

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def generate_hotspot_overlay(image_bytes: bytes) -> tuple[bytes, dict]:
    """
    Generate a pseudo-color heatmap overlay to visualise thermal hotspots.
    Uses OpenCV's COLORMAP_JET applied to a grayscale version of the image.
    Replace this with your actual DL model inference when available.
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode thermal image.")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Threshold to find bright (hot) regions
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.4, heatmap, 0.6, 0)

    hotspots = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 255), 2)
        hotspots.append({"bbox": [x, y, x + w, y + h], "area_px": int(area)})

    _, buf = cv2.imencode(".jpg", overlay)
    return buf.tobytes(), {
        "hotspot_count": len(hotspots),
        "hotspots": hotspots,
        "note": "Replace with DL model for production accuracy.",
    }