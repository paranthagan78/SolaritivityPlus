"""modules/llm_summary/summary_engine.py
Gemini-based summary generation using Google AI Studio API key.
Uses per-image rows from detections.csv and carbon.csv.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import pandas as pd
import requests

from config import DETECTIONS_CSV, CARBON_CSV
from .prompt_templates import SYSTEM_INSTRUCTION, USER_PROMPT_TEMPLATE

# REPLACE with a function so key is always read fresh:
def _get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "Missing GEMINI_API_KEY (or GOOGLE_API_KEY) in environment. "
            "Add it to your .env file without spaces: GEMINI_API_KEY=AIza..."
        )
    return key

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "").strip()
GEMINI_TIMEOUT_SEC = int(os.environ.get("GEMINI_TIMEOUT_SEC", "90"))

_PREFERRED_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
    "gemini-1.5-pro-latest",
]


def _read_csv_safe(path: str) -> pd.DataFrame:
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def _sanitize_records(df: pd.DataFrame, max_rows: int = 30) -> List[Dict]:
    if df.empty:
        return []
    tail_df = df.tail(max_rows).copy()
    tail_df = tail_df.where(pd.notna(tail_df), None)
    return tail_df.to_dict(orient="records")


def _numeric_summary(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {}
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return {}
    out = {}
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        out[col] = {
            "count": int(s.shape[0]),
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "median": float(s.median()),
        }
    return out


def _normalize_filename(v: str) -> str:
    return os.path.basename(str(v)).strip().lower()


def _filter_by_image(df: pd.DataFrame, filename: str | None) -> Tuple[pd.DataFrame, str]:
    if df.empty:
        return df, filename or "unknown"

    if "image_filename" not in df.columns:
        if filename:
            return df, filename
        return df.tail(1), "latest_record"

    work = df.copy()
    work["_norm_image"] = work["image_filename"].astype(str).map(_normalize_filename)

    if filename and filename.strip():
        target = _normalize_filename(filename)
        exact = work[work["_norm_image"] == target]
        if not exact.empty:
            return exact.drop(columns=["_norm_image"]), target

        partial = work[work["_norm_image"].str.contains(target, na=False)]
        if not partial.empty:
            return partial.drop(columns=["_norm_image"]), target

        return work.iloc[0:0].drop(columns=["_norm_image"]), target

    latest = work.tail(1)
    resolved = str(latest["image_filename"].iloc[0])
    return latest.drop(columns=["_norm_image"]), resolved


def list_available_models() -> List[str]:
    api_key = _get_api_key()
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    resp = requests.get(url, timeout=GEMINI_TIMEOUT_SEC)

    if resp.status_code == 403:
        error_body = resp.json().get("error", {})
        raise ValueError(
            f"API key rejected (403): {error_body.get('message', resp.text)}\n"
            f"Your key may be leaked or invalid. Generate a new one at https://aistudio.google.com"
        )
    if resp.status_code != 200:
        raise ValueError(f"Failed to list models: {resp.status_code} {resp.text}")

    data = resp.json()
    names = []
    for m in data.get("models", []):
        if "generateContent" in m.get("supportedGenerationMethods", []):
            names.append(m.get("name", "").replace("models/", ""))
    return names


def _resolve_model() -> str:
    available = list_available_models()

    if not available:
        raise ValueError(
            "No generateContent-capable Gemini models found for this API key. "
            "Check your key at https://aistudio.google.com"
        )

    if GEMINI_MODEL:
        if GEMINI_MODEL in available:
            return GEMINI_MODEL
        raise ValueError(
            f"Configured GEMINI_MODEL '{GEMINI_MODEL}' is not available for your key.\n"
            f"Available models for your key: {available}"
        )

    for m in _PREFERRED_MODELS:
        if m in available:
            return m

    # Use first available if none of preferred match
    return available[0]


def _call_gemini(system_instruction: str, user_prompt: str) -> str:
    api_key = _get_api_key()
    model_name = _resolve_model()
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={api_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 4096,
        },
    }

    resp = requests.post(url, json=payload, timeout=GEMINI_TIMEOUT_SEC)

    if resp.status_code == 403:
        error_body = resp.json().get("error", {})
        raise ValueError(
            f"Gemini API key rejected (403): {error_body.get('message', resp.text)}\n"
            f"Generate a new key at https://aistudio.google.com"
        )
    if resp.status_code != 200:
        raise ValueError(f"Gemini API error {resp.status_code}: {resp.text}")

    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError("Gemini returned no candidates.")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        raise ValueError("Gemini returned empty text.")
    return text


def generate_summary(filename_filter: str | None = None) -> str:
    det_df = _read_csv_safe(DETECTIONS_CSV)
    car_df = _read_csv_safe(CARBON_CSV)

    if det_df.empty and car_df.empty:
        raise ValueError("No data found in detections.csv or carbon.csv.")

    det_filtered, resolved_image = _filter_by_image(det_df, filename_filter)
    car_filtered, _ = _filter_by_image(
        car_df, resolved_image if not filename_filter else filename_filter
    )

    if filename_filter and det_filtered.empty and car_filtered.empty:
        raise ValueError(f"No records found for image filename filter: {filename_filter}")

    detection_rows = _sanitize_records(det_filtered)
    carbon_rows = _sanitize_records(car_filtered)

    prompt = USER_PROMPT_TEMPLATE.format(
        target_image=resolved_image,
        detection_rows_json=json.dumps(detection_rows, indent=2, ensure_ascii=True),
        carbon_rows_json=json.dumps(carbon_rows, indent=2, ensure_ascii=True),
        detection_numeric_json=json.dumps(_numeric_summary(det_filtered), indent=2, ensure_ascii=True),
        carbon_numeric_json=json.dumps(_numeric_summary(car_filtered), indent=2, ensure_ascii=True),
    )

    return _call_gemini(SYSTEM_INSTRUCTION, prompt)