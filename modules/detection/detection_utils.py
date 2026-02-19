"""modules/detection/detection_utils.py"""
from PIL import Image, ImageDraw, ImageFont

COLOR_MAP = {
    "black_core": "#e74c3c",
    "crack":      "#e67e22",
    "finger":     "#f1c40f",
    "star_crack": "#9b59b6",
    "thick_line": "#1abc9c",
}


def draw_boxes(image_path: str, detections: list, out_path: str) -> str:
    """Draw bounding boxes on image and save to out_path."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", max(14, img.width // 50))
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = COLOR_MAP.get(det["class_name"], "#00ff00")
        for i in range(3):
            draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)
        label = f"{det['class_name']} {det['confidence']:.0%}"
        try:
            bb = draw.textbbox((x1, y1 - 20), label, font=font)
            draw.rectangle([bb[0] - 2, bb[1] - 2, bb[2] + 2, bb[3] + 2], fill=color)
            draw.text((x1, y1 - 20), label, fill="white", font=font)
        except Exception:
            draw.text((x1, max(0, y1 - 20)), label, fill=color, font=font)

    img.save(out_path)
    return out_path


def compute_area_ratios(detections: list, img_w: int, img_h: int) -> list:
    """Add area_ratio field to each detection in-place."""
    panel_area = img_w * img_h if img_w * img_h > 0 else 1
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        det["area_ratio"] = round((x2 - x1) * (y2 - y1) / panel_area, 6)
    return detections