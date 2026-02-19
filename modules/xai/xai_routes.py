"""modules/xai/xai_routes.py"""
import os, uuid
from flask import Blueprint, request, jsonify
from auth import require_auth
from config import UPLOAD_FOLDER, EXPLAIN_FOLDER
from modules.el_upload.upload_utils import allowed_file
from .gradcam import generate_gradcam

xai_bp = Blueprint("xai", __name__, url_prefix="/api/xai")

@xai_bp.route("/gradcam", methods=["POST"])
@require_auth
def gradcam():
    file = request.files.get("image")
    if not file or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Valid image required."}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    uid = uuid.uuid4().hex
    img_path = os.path.join(UPLOAD_FOLDER, f"xai_{uid}.{ext}")
    file.save(img_path)

    out_name = f"gradcam_{uid}.jpg"
    out_path = os.path.join(EXPLAIN_FOLDER, out_name)

    try:
        generate_gradcam(img_path, out_path)
        return jsonify({
            "success": True,
            "explanation_image": f"/explanations/{out_name}",
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500