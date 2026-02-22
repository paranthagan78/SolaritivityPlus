"""modules/thermal/thermal_routes.py"""
import os, uuid
from flask import Blueprint, request, jsonify
from PIL import Image
from auth import require_auth
from config import THERMAL_UPLOAD_FOLDER, RESULT_FOLDER, UPLOAD_FOLDER, EXPLAIN_FOLDER
from modules.el_upload.upload_utils import allowed_file
from .thermal_model import generate_hotspot_overlay
from .thermal_detect_model import run_thermal_detection

thermal_bp = Blueprint("thermal", __name__, url_prefix="/api/thermal")


# ── Original hotspot overlay route ─────────────────────────────────────────
@thermal_bp.route("/predict", methods=["POST"])
@require_auth
def predict():
    file = request.files.get("image")
    if not file or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Valid thermal image required."}), 400

    image_bytes = file.read()
    uid = uuid.uuid4().hex
    ext = file.filename.rsplit(".", 1)[1].lower()

    orig_path = os.path.join(THERMAL_UPLOAD_FOLDER, f"{uid}.{ext}")
    with open(orig_path, "wb") as fp:
        fp.write(image_bytes)

    try:
        overlay_bytes, info = generate_hotspot_overlay(image_bytes)
        result_name = f"thermal_{uid}.jpg"
        result_path = os.path.join(RESULT_FOLDER, result_name)
        with open(result_path, "wb") as fp:
            fp.write(overlay_bytes)

        return jsonify({
            "success": True,
            "result_image": f"/results/{result_name}",
            **info,
        }), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── YOLO thermal detection ─────────────────────────────────────────────────
@thermal_bp.route("/detect", methods=["POST"])
@require_auth
def detect():
    file = request.files.get("image")
    if not file or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Valid image required."}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    uid = uuid.uuid4().hex
    img_path = os.path.join(THERMAL_UPLOAD_FOLDER, f"{uid}.{ext}")
    file.save(img_path)

    try:
        result = run_thermal_detection(img_path)
        img = Image.open(img_path)
        w, h = img.size

        dets = result["detections"]
        # Compute area ratios
        panel_area = w * h if w * h > 0 else 1
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            d["area_ratio"] = round((x2 - x1) * (y2 - y1) / panel_area, 6)

        # Draw boxes and save result image
        from modules.detection.detection_utils import draw_boxes
        result_name = f"thdet_{uid}.{ext}"
        result_path = os.path.join(RESULT_FOLDER, result_name)
        draw_boxes(img_path, dets, result_path)

        return jsonify({
            "success": True,
            "filename": f"{uid}.{ext}",
            "detections": dets,
            "count": len(dets),
            "result_image": f"/results/{result_name}",
            "image_size": {"width": w, "height": h},
        }), 200

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ── Thermal LIME ────────────────────────────────────────────────────────
@thermal_bp.route("/lime", methods=["POST"])
@require_auth
def thermal_lime():
    file = request.files.get("image")
    if not file or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Valid image required."}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    uid = uuid.uuid4().hex
    img_path = os.path.join(UPLOAD_FOLDER, f"thxai_{uid}.{ext}")
    file.save(img_path)

    out_name = f"thlime_{uid}.jpg"
    out_path = os.path.join(EXPLAIN_FOLDER, out_name)

    try:
        from modules.xai.lime_xai import generate_lime
        generate_lime(img_path, out_path)
        return jsonify({
            "success": True,
            "explanation_image": f"/explanations/{out_name}",
        }), 200
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500