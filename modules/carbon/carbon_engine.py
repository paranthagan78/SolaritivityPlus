"""modules/carbon/carbon_engine.py"""
import joblib, numpy as np
from config import CARBON_MODEL_PATH, LABEL_ENCODER_PATH, INDIA_EMISSION_FACTORS, DETECTION_CLASSES

_carbon_model = None
_label_encoder = None

def _load():
    global _carbon_model, _label_encoder
    if _carbon_model is None:
        _carbon_model = joblib.load(CARBON_MODEL_PATH)
        _label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return _carbon_model, _label_encoder

def predict_carbon(detections: list, img_w: int, img_h: int,
                   city: str = "Chennai", panel_power: float = 380,
                   ambient_temp: float = 32, irradiance: float = 900) -> dict:
    model, le = _load()
    emission_factor = INDIA_EMISSION_FACTORS.get(city, 0.80)

    area_ratios, defect_types = [], []
    for det in detections:
        if det["confidence"] >= 0.5:
            area_ratios.append(det.get("area_ratio", 0))
            defect_types.append(det["class_name"])

    num_defects = len(area_ratios)
    total_degradation = min(sum(area_ratios) * 0.5, 0.4) if area_ratios else 0.0
    dominant = max(set(defect_types), key=defect_types.count) if defect_types else list(DETECTION_CLASSES.values())[0]

    defect_encoded = le.transform([dominant])[0]
    X = np.array([[num_defects, panel_power, ambient_temp, irradiance, emission_factor, total_degradation, defect_encoded]])
    co2 = float(model.predict(X)[0])

    return {
        "co2_kg_per_year": round(co2, 2),
        "num_defects": num_defects,
        "total_degradation_pct": round(total_degradation * 100, 2),
        "dominant_defect": dominant,
        "city": city,
        "emission_factor": emission_factor,
        "panel_power_w": panel_power,
        "ambient_temp_c": ambient_temp,
        "irradiance_w_m2": irradiance,
    }