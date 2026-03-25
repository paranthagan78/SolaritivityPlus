"""modules/chatbot/chatbot_routes.py
Flask blueprint for the RAG chatbot — with sentiment analysis support.

CHANGES FROM ORIGINAL:
  - /api/chat/query now returns sentiment metadata (label, score, compound, is_negative)
    alongside the answer, so the frontend can show a mood indicator or adapt the UI.
  - answer_query() now returns a dict instead of a plain string.
  - All existing routes and their signatures are preserved unchanged.
"""
from flask import Blueprint, request, jsonify
from auth import require_auth
from .rag_engine import answer_query
from .vector_store import ingest_docs, ingest_csvs, get_stats, ingest_image_result

chatbot_bp = Blueprint("chatbot", __name__, url_prefix="/api/chat")


@chatbot_bp.route("/query", methods=["POST"])
@require_auth
def query():
    """
    POST body:
    {
        "question":       "What defects were found?",
        "history":        [{role, content}, ...],   // optional
        "image_filename": "abc123.jpg"              // optional, current session image
    }

    Response body:
    {
        "success":            true,
        "answer":             "...",
        "sentiment_label":    "negative",      // "positive" | "neutral" | "negative"
        "sentiment_score":    0.91,            // DistilBERT confidence
        "sentiment_compound": -0.82,           // -1.0 (very negative) → +1.0 (very positive)
        "is_negative":        true             // true = empathetic mode was activated
    }
    """
    body           = request.get_json(silent=True) or {}
    question       = (body.get("question") or "").strip()
    history        = body.get("history", [])
    image_filename = (body.get("image_filename") or "").strip() or None

    if not question:
        return jsonify({"success": False, "error": "Question is required."}), 400

    try:
        result = answer_query(question, history, image_filename=image_filename)

        return jsonify({
            "success":            True,
            "answer":             result["answer"],
            "sentiment_label":    result["sentiment_label"],
            "sentiment_score":    result["sentiment_score"],
            "sentiment_compound": result["sentiment_compound"],
            "is_negative":        result["is_negative"],
        }), 200

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@chatbot_bp.route("/ingest", methods=["POST"])
@require_auth
def ingest():
    """Re-ingest docs and CSVs into ChromaDB on demand."""
    try:
        doc_count = ingest_docs()
        csv_count = ingest_csvs()
        return jsonify({
            "success":    True,
            "message":    "Ingestion complete.",
            "doc_chunks": doc_count,
            "csv_rows":   csv_count,
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@chatbot_bp.route("/ingest_image", methods=["POST"])
@require_auth
def ingest_image():
    """
    Called automatically after detection/carbon pipeline runs for an image.
    POST body:
    {
        "image_filename": "abc123.jpg",
        "detections": [
            {
                "defect_class": "crack",
                "confidence": 0.82,
                "area_ratio": 0.042,
                "bbox_x1": 526, "bbox_y1": 419,
                "bbox_x2": 869, "bbox_y2": 549,
                "image_width": 1024, "image_height": 1024
            },
            ...
        ],
        "carbon_data": {
            "city": "Delhi",
            "panel_power_w": 380,
            "ambient_temp_c": 25,
            "irradiance_w_m2": 900,
            "emission_factor": 0.85,
            "num_defects": 2,
            "dominant_defect": "crack",
            "total_degradation_pct": 1.42,
            "co2_kg_per_year": 9.55
        }
    }
    """
    body           = request.get_json(silent=True) or {}
    image_filename = (body.get("image_filename") or "").strip()
    detections     = body.get("detections", [])
    carbon_data    = body.get("carbon_data", None)

    if not image_filename:
        return jsonify({"success": False, "error": "image_filename is required."}), 400

    try:
        ingest_image_result(image_filename, detections, carbon_data)
        return jsonify({
            "success": True,
            "message": f"Ingested results for {image_filename}",
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@chatbot_bp.route("/stats", methods=["GET"])
@require_auth
def stats():
    """Return vector store stats for debugging."""
    return jsonify({"success": True, **get_stats()}), 200