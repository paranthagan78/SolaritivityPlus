"""modules/integrity/integrity_routes.py"""
from flask import Blueprint, request, jsonify
from auth import require_auth
from .integrity_model import predict_integrity
from modules.el_upload.upload_utils import allowed_file

integrity_bp = Blueprint("integrity", __name__, url_prefix="/api/integrity")

@integrity_bp.route("/check", methods=["POST"])
@require_auth
def check():
    """Check one or many images for integrity (valid EL image vs noise/wrong image)."""
    files = request.files.getlist("images")
    if not files:
        return jsonify({"success": False, "error": "No images provided."}), 400

    results = []
    for f in files:
        if not allowed_file(f.filename):
            results.append({"filename": f.filename, "error": "unsupported type"})
            continue
        try:
            data = f.read()
            res = predict_integrity(data)
            res["filename"] = f.filename
            results.append(res)
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})

    return jsonify({"success": True, "results": results}), 200