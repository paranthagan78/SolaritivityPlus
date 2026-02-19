"""modules/detection/csv_writer.py"""
import csv, os
from datetime import datetime
from config import DETECTIONS_CSV

HEADERS = [
    "timestamp", "image_filename", "defect_class", "confidence",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "area_ratio", "image_width", "image_height",
]

def write_detections(filename: str, detections: list, img_w: int, img_h: int):
    file_exists = os.path.isfile(DETECTIONS_CSV)
    with open(DETECTIONS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        if not file_exists:
            writer.writeheader()
        ts = datetime.now().isoformat()
        if not detections:
            writer.writerow({
                "timestamp": ts, "image_filename": filename,
                "defect_class": "none", "confidence": 0,
                "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 0, "bbox_y2": 0,
                "area_ratio": 0, "image_width": img_w, "image_height": img_h,
            })
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            writer.writerow({
                "timestamp": ts,
                "image_filename": filename,
                "defect_class": det["class_name"],
                "confidence": det["confidence"],
                "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                "area_ratio": det.get("area_ratio", 0),
                "image_width": img_w,
                "image_height": img_h,
            })