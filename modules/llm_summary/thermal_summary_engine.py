"""modules/llm_summary/thermal_summary_engine.py
Gemini-based thermal summary generation.
"""
from __future__ import annotations
import json
import os
from typing import Dict, List
from .summary_engine import _call_gemini
from .thermal_prompt_templates import SYSTEM_INSTRUCTION_THERMAL, USER_PROMPT_THERMAL_TEMPLATE

def generate_thermal_summary(image_filename: str, detections: List[Dict], lime_features: List[Dict] = None) -> str:
    """
    Generate a thermal report using LLM.
    """
    if not detections and not lime_features:
        raise ValueError("No thermal detection or XAI data provided for summary.")

    prompt = USER_PROMPT_THERMAL_TEMPLATE.format(
        target_image=image_filename,
        thermal_detections_json=json.dumps(detections, indent=2),
        lime_features_json=json.dumps(lime_features or [], indent=2)
    )

    return _call_gemini(SYSTEM_INSTRUCTION_THERMAL, prompt)
