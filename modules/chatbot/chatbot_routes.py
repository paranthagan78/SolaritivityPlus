"""modules/chatbot/chatbot_routes.py"""
from flask import Blueprint, request, jsonify
from auth import require_auth
from .rag_engine import answer_query
from .vector_store import ingest_docs, ingest_csvs

chatbot_bp = Blueprint("chatbot", __name__, url_prefix="/api/chat")

@chatbot_bp.route("/query", methods=["POST"])
@require_auth
def query():
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    history  = body.get("history", [])   # [{role, content}, ...]

    if not question:
        return jsonify({"success": False, "error": "Question is required."}), 400

    try:
        answer = answer_query(question, history)
        return jsonify({"success": True, "answer": answer}), 200
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@chatbot_bp.route("/ingest", methods=["POST"])
@require_auth
def ingest():
    """Re-ingest docs and CSVs into ChromaDB on demand."""
    try:
        ingest_docs()
        ingest_csvs()
        return jsonify({"success": True, "message": "Ingestion complete."}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500