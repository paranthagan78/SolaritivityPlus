"""modules/detection/detection_routes.py"""
import os, uuid
from flask import Blueprint, request, jsonify, send_from_directory
from PIL import Image
from auth import require_auth
from config import UPLOAD_FOLDER, RESULT_FOLDER
from modules.el_upload.upload_utils import allowed_file
from .detection_model import run_detection
from .detection_utils import draw_boxes, compute_area_ratios
from .csv_writer import write_detections

detection_bp = Blueprint("detection", __name__, url_prefix="/api/detect")

@detection_bp.route("/run", methods=["POST"])
@require_auth
def run():
    file = request.files.get("image")
    if not file or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Valid image required."}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    uid = uuid.uuid4().hex
    img_path = os.path.join(UPLOAD_FOLDER, f"{uid}.{ext}")
    file.save(img_path)

    try:
        result = run_detection(img_path)
        img = Image.open(img_path)
        w, h = img.size

        dets = compute_area_ratios(result["detections"], w, h)

        result_name = f"det_{uid}.{ext}"
        result_path = os.path.join(RESULT_FOLDER, result_name)
        draw_boxes(img_path, dets, result_path)

        write_detections(f"{uid}.{ext}", dets, w, h)

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