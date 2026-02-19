"""modules/thermal/thermal_routes.py"""
import os, uuid
from flask import Blueprint, request, jsonify
from auth import require_auth
from config import THERMAL_UPLOAD_FOLDER, RESULT_FOLDER
from modules.el_upload.upload_utils import allowed_file
from .thermal_model import generate_hotspot_overlay

thermal_bp = Blueprint("thermal", __name__, url_prefix="/api/thermal")

@thermal_bp.route("/predict", methods=["POST"])
@require_auth
def predict():
    file = request.files.get("image")
    if not file or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Valid thermal image required."}), 400

    image_bytes = file.read()
    uid = uuid.uuid4().hex
    ext = file.filename.rsplit(".", 1)[1].lower()

    # Save original
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