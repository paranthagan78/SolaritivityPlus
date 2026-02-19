"""modules/llm_summary/summary_routes.py"""
from flask import Blueprint, request, jsonify
from auth import require_auth
from .summary_engine import generate_summary

llm_bp = Blueprint("llm_summary", __name__, url_prefix="/api/summary")

@llm_bp.route("/generate", methods=["POST"])
@require_auth
def generate():
    body = request.get_json(silent=True) or {}
    filename_filter = body.get("filename")
    try:
        text = generate_summary(filename_filter)
        return jsonify({"success": True, "summary": text}), 200
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500