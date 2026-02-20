"""modules/weather/weather_routes.py"""
import requests
from flask import Blueprint, request, jsonify

weather_bp = Blueprint("weather", __name__, url_prefix="/api/weather")
TIMEOUT = 5

GEO_API_URL     = "http://ip-api.com/json/"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"


@weather_bp.route("/location", methods=["GET"])
def get_location():
    try:
        ip = request.headers.get("X-Forwarded-For", request.remote_addr)
        ip = ip.split(",")[0].strip().split(":")[0]
        r    = requests.get(f"{GEO_API_URL}{ip}", timeout=TIMEOUT)
        data = r.json()
        if data.get("status") != "success":
            # Fallback for localhost / private IPs
            return jsonify({
                "success": True,
                "city": "Thalavapalayam, Karur", "region": "Tamil Nadu", "country": "India",
                "lat": 11.0630, "lon": 78.0466,
                "note": "Fallback location (localhost detected)",
            }), 200
        return jsonify({
            "success": True,
            "city":    data.get("city", "Unknown"),
            "region":  data.get("regionName", ""),
            "country": data.get("country", ""),
            "lat":     data.get("lat"),
            "lon":     data.get("lon"),
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@weather_bp.route("/temperature", methods=["GET"])
def get_temperature():
    try:
        lat = request.args.get("lat", type=float)
        lon = request.args.get("lon", type=float)
        if lat is None or lon is None:
            return jsonify({"success": False, "error": "lat and lon required."}), 400
        r    = requests.get(WEATHER_API_URL,
                            params={"latitude": lat, "longitude": lon,
                                    "current": "temperature_2m", "timezone": "auto"},
                            timeout=TIMEOUT)
        temp = r.json()["current"]["temperature_2m"]
        return jsonify({"success": True, "temperature_c": temp}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500