"""modules/thermal/thermal_detect_model.py — YOLO detection on thermal images."""
from ultralytics import YOLO
from config import THERMAL_YOLO_MODEL_PATH

_model = None


def _load():
    global _model
    if _model is None:
        _model = YOLO(THERMAL_YOLO_MODEL_PATH)
    return _model


def run_thermal_detection(image_path: str, conf: float = 0.3) -> dict:
    model = _load()
    results = model.predict(source=image_path, conf=conf, save=False, verbose=False)
    r = results[0]

    # Read class names from the model itself
    names = r.names  # {0: 'class_a', 1: 'class_b', ...}

    detections = []
    boxes = r.boxes
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = [round(float(v)) for v in box.xyxy[0]]
            detections.append({
                "class_id": cls_id,
                "class_name": names.get(cls_id, f"class_{cls_id}"),
                "confidence": round(conf_val, 4),
                "bbox": [x1, y1, x2, y2],
            })

    return {
        "detections": detections,
        "count": len(detections),
        "image_shape": list(r.orig_shape),
    }
