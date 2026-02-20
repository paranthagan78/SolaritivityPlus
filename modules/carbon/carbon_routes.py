"""modules/carbon/carbon_routes.py"""
from flask import Blueprint, request, jsonify
from auth import require_auth
from .carbon_engine import predict_carbon
from .csv_writer import write_carbon

carbon_bp = Blueprint("carbon", __name__, url_prefix="/api/carbon")

@carbon_bp.route("/predict", methods=["POST"])
@require_auth
def predict():
    body         = request.get_json(silent=True) or {}
    detections   = body.get("detections", [])
    img_w        = body.get("image_width", 640)
    img_h        = body.get("image_height", 640)
    city         = body.get("city", "Chennai")
    panel_power  = float(body.get("panel_power", 380))
    ambient_temp = float(body.get("ambient_temp", 32))
    irradiance   = float(body.get("irradiance", 900))
    filename     = body.get("filename", "unknown")

    try:
        result = predict_carbon(
            detections, img_w, img_h,
            city, panel_power, ambient_temp, irradiance,
        )
        write_carbon(filename, result)

        # ── Ingest into ChromaDB so chatbot can answer image-specific questions ──
        try:
            from modules.chatbot.vector_store import ingest_image_result

            # Normalise detections to the shape ingest_image_result expects.
            # Detection dicts from the frontend use "class_name"; handle both forms.
            normalised = [
                {
                    "defect_class": d.get("class_name", d.get("defect_class", "unknown")),
                    "confidence":   d.get("confidence", 0),
                    "area_ratio":   d.get("area_ratio", 0),
                    "bbox_x1":      d.get("bbox_x1", d.get("x1", 0)),
                    "bbox_y1":      d.get("bbox_y1", d.get("y1", 0)),
                    "bbox_x2":      d.get("bbox_x2", d.get("x2", 0)),
                    "bbox_y2":      d.get("bbox_y2", d.get("y2", 0)),
                    "image_width":  img_w,
                    "image_height": img_h,
                }
                for d in detections
                if d.get("confidence", 0) >= 0.5
            ]

            ingest_image_result(
                image_filename=filename,
                detections=normalised,
                carbon_data=result,   # full dict from predict_carbon
            )
        except Exception as rag_err:
            # Never let RAG errors break the main carbon response
            print(f"[Carbon] RAG ingest warning: {rag_err}")

        return jsonify({"success": True, **result}), 200

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500