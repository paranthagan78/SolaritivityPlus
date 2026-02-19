"""modules/carbon/carbon_routes.py"""
from flask import Blueprint, request, jsonify
from auth import require_auth
from .carbon_engine import predict_carbon
from .csv_writer import write_carbon

carbon_bp = Blueprint("carbon", __name__, url_prefix="/api/carbon")

@carbon_bp.route("/predict", methods=["POST"])
@require_auth
def predict():
    body = request.get_json(silent=True) or {}
    detections = body.get("detections", [])
    img_w = body.get("image_width", 640)
    img_h = body.get("image_height", 640)
    city = body.get("city", "Chennai")
    panel_power = float(body.get("panel_power", 380))
    ambient_temp = float(body.get("ambient_temp", 32))
    irradiance = float(body.get("irradiance", 900))
    filename = body.get("filename", "unknown")

    try:
        result = predict_carbon(detections, img_w, img_h, city, panel_power, ambient_temp, irradiance)
        write_carbon(filename, result)
        return jsonify({"success": True, **result}), 200
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500