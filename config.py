"""config.py — Single source of truth for all paths and constants."""
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Folders ────────────────────────────────────────────────────────────────
UPLOAD_FOLDER         = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER         = os.path.join(BASE_DIR, "results")
THERMAL_UPLOAD_FOLDER = os.path.join(BASE_DIR, "thermal_uploads")
DATA_FOLDER           = os.path.join(BASE_DIR, "data")
DOCS_FOLDER           = os.path.join(BASE_DIR, "docs")
CHROMA_FOLDER         = os.path.join(BASE_DIR, "chroma_db")
EXPLAIN_FOLDER        = os.path.join(BASE_DIR, "explanations")

for _d in [UPLOAD_FOLDER, RESULT_FOLDER, THERMAL_UPLOAD_FOLDER,
           DATA_FOLDER, DOCS_FOLDER, CHROMA_FOLDER, EXPLAIN_FOLDER]:
    os.makedirs(_d, exist_ok=True)

# ── Model paths ────────────────────────────────────────────────────────────
YOLO_MODEL_PATH          = os.path.join(BASE_DIR, "model", "best.pt")
THERMAL_YOLO_MODEL_PATH  = os.path.join(BASE_DIR, "model", "bestthermal.pt")
INTEGRITY_MODEL_PATH     = os.path.join(BASE_DIR, "model", "integrity_model.pth")
CARBON_MODEL_PATH        = os.path.join(BASE_DIR, "model", "carbon_model.pkl")
LABEL_ENCODER_PATH       = os.path.join(BASE_DIR, "model", "label_encoder.pkl")

# ── CSV paths ──────────────────────────────────────────────────────────────
DETECTIONS_CSV = os.path.join(DATA_FOLDER, "detections.csv")
CARBON_CSV     = os.path.join(DATA_FOLDER, "carbon.csv")

# ── Image settings ─────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS  = {"png", "jpg", "jpeg", "bmp", "tiff"}
MAX_CONTENT_LENGTH  = 50 * 1024 * 1024   # 50 MB

# ── Detection classes ──────────────────────────────────────────────────────
DETECTION_CLASSES = {
    0: "black_core",
    1: "crack",
    2: "finger",
    3: "star_crack",
    4: "thick_line",
}

# ── Carbon / city emission factors ─────────────────────────────────────────
INDIA_EMISSION_FACTORS = {
    "Chennai":   0.82,
    "Mumbai":    0.78,
    "Delhi":     0.85,
    "Bangalore": 0.75,
    "Hyderabad": 0.80,
}

# ── Auth ───────────────────────────────────────────────────────────────────
PASSCODE         = os.environ.get("PASSCODE", "SOLAR@2025")
SESSION_TIMEOUT  = int(os.environ.get("SESSION_TIMEOUT", 3600))
MAX_ATTEMPTS     = 5
LOCKOUT_DURATION = 300

# ── Local LLM (GGUF via ctransformers) ────────────────────────────────────
LOCAL_LLM_PATH       = os.environ.get(
    "LOCAL_LLM_PATH",
    r"C:\Users\paran\OneDrive\Desktop\Fun Projects\chat_v\orca-mini-3b.q4_0.gguf"
)
LOCAL_LLM_TYPE       = "llama"
LOCAL_LLM_MAX_TOKENS = int(os.environ.get("LOCAL_LLM_MAX_TOKENS", 512))
LOCAL_LLM_TEMP       = float(os.environ.get("LOCAL_LLM_TEMP", 0.7))

# ── Weather / Geo (free, no API key) ──────────────────────────────────────
GEO_API_URL     = "http://ip-api.com/json/"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"