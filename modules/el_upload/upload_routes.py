"""modules/el_upload/upload_routes.py"""
from flask import Blueprint, request, jsonify
from auth import require_auth
from .upload_utils import allowed_file, save_upload, get_file_info

el_bp = Blueprint("el_upload", __name__, url_prefix="/api/el")

@el_bp.route("/upload", methods=["POST"])
@require_auth
def upload():
    """Upload one or many EL images."""
    files = request.files.getlist("images")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"success": False, "error": "No files provided."}), 400

    saved, errors = [], []
    for file in files:
        if not allowed_file(file.filename):
            errors.append(f"{file.filename}: unsupported type")
            continue
        try:
            fname, path, orig = save_upload(file)
            saved.append(get_file_info(fname, orig, path))
        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")

    if not saved:
        return jsonify({"success": False, "error": "No valid files uploaded.", "errors": errors}), 400

    return jsonify({"success": True, "uploaded": saved, "count": len(saved), "errors": errors}), 200