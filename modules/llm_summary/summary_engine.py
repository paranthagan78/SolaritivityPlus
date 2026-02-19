"""modules/llm_summary/summary_engine.py
Uses local GGUF model (orca-mini-3b) via ctransformers for report generation.
"""
import re
import pandas as pd
from ctransformers import AutoModelForCausalLM
from config import (LOCAL_LLM_PATH, LOCAL_LLM_TYPE,
                    LOCAL_LLM_MAX_TOKENS, LOCAL_LLM_TEMP,
                    DETECTIONS_CSV, CARBON_CSV)
from .prompt_templates import SUMMARY_PROMPT

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        print(f"[LLM] Loading local model from: {LOCAL_LLM_PATH}")
        _llm = AutoModelForCausalLM.from_pretrained(
            LOCAL_LLM_PATH,
            model_type=LOCAL_LLM_TYPE,
            max_new_tokens=LOCAL_LLM_MAX_TOKENS,
            temperature=LOCAL_LLM_TEMP,
            local_files_only=True,
        )
        print("[LLM] Model loaded.")
    return _llm


def _preprocess(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9.,!?%:/()\-\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _read_csv(path: str, max_rows: int = 25) -> str:
    try:
        df = pd.read_csv(path)
        return df.tail(max_rows).to_string(index=False)
    except Exception:
        return "No data available."


def _build_prompt(det_text: str, carbon_text: str) -> str:
    system = (
        "You are a solar PV inspection analyst. "
        "Read the defect detection and carbon emission data below, "
        "then write a concise structured report."
    )
    data_block = (
        f"=== DETECTION DATA ===\n{_preprocess(det_text)}\n\n"
        f"=== CARBON EMISSION DATA ===\n{_preprocess(carbon_text)}"
    )
    instruction = SUMMARY_PROMPT.format(
        detection_data=_preprocess(det_text),
        carbon_data=_preprocess(carbon_text),
    )
    prompt = (
        f"### System:\n{system}\n\n"
        f"### User:\n{instruction}\n\n"
        f"### Response:\n"
    )
    return prompt


def generate_summary(filename_filter: str = None) -> str:
    det_text    = _read_csv(DETECTIONS_CSV)
    carbon_text = _read_csv(CARBON_CSV)

    if filename_filter:
        try:
            d = pd.read_csv(DETECTIONS_CSV)
            d = d[d["image_filename"].str.contains(filename_filter, na=False)]
            if not d.empty:
                det_text = d.to_string(index=False)

            c = pd.read_csv(CARBON_CSV)
            c = c[c["image_filename"].str.contains(filename_filter, na=False)]
            if not c.empty:
                carbon_text = c.to_string(index=False)
        except Exception:
            pass

    prompt = _build_prompt(det_text, carbon_text)
    llm    = _get_llm()

    response = ""
    for token in llm(prompt, stream=True):
        response += token

    return response.strip()