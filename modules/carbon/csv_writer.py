"""modules/carbon/csv_writer.py"""
import csv, os
from datetime import datetime
from config import CARBON_CSV

HEADERS = [
    "timestamp", "image_filename", "city", "panel_power_w",
    "ambient_temp_c", "irradiance_w_m2", "emission_factor",
    "num_defects", "dominant_defect", "total_degradation_pct",
    "co2_kg_per_year",
]

def write_carbon(filename: str, data: dict):
    exists = os.path.isfile(CARBON_CSV)
    with open(CARBON_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADERS)
        if not exists:
            w.writeheader()
        w.writerow({
            "timestamp": datetime.now().isoformat(),
            "image_filename": filename,
            "city": data.get("city"),
            "panel_power_w": data.get("panel_power_w"),
            "ambient_temp_c": data.get("ambient_temp_c"),
            "irradiance_w_m2": data.get("irradiance_w_m2"),
            "emission_factor": data.get("emission_factor"),
            "num_defects": data.get("num_defects"),
            "dominant_defect": data.get("dominant_defect"),
            "total_degradation_pct": data.get("total_degradation_pct"),
            "co2_kg_per_year": data.get("co2_kg_per_year"),
        })