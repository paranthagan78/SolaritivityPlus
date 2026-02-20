"""modules/chatbot/rag_engine.py
RAG chatbot using Gemini API.
Retrieves context from ChromaDB (cosine similarity), with smart image-context
injection when the user asks about a specific uploaded image.
"""
import os
import re
import time
import requests
from dotenv import load_dotenv
load_dotenv()

from .vector_store import query_collection, query_collection_with_filter

# ── Gemini config ─────────────────────────────────────────────────────────
GEMINI_TIMEOUT_SEC = int(os.environ.get("GEMINI_TIMEOUT_SEC", "90"))

_PREFERRED_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
]

_resolved_model: str | None = None


def _get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "Missing GEMINI_API_KEY in environment. "
            "Add it to your .env file: GEMINI_API_KEY=AIza..."
        )
    return key


def _list_available_models(api_key: str) -> list:
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return [
            m["name"].replace("models/", "")
            for m in data.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]
    except Exception as e:
        print(f"[RAG] Could not list models: {e}")
        return []


def _resolve_model(api_key: str) -> str:
    global _resolved_model
    if _resolved_model:
        return _resolved_model

    env_model = os.environ.get("GEMINI_MODEL", "").strip()
    if env_model:
        _resolved_model = env_model
        print(f"[RAG] Using env-specified model: {_resolved_model}")
        return _resolved_model

    available = _list_available_models(api_key)
    for preferred in _PREFERRED_MODELS:
        if preferred in available:
            _resolved_model = preferred
            print(f"[RAG] Auto-selected model: {_resolved_model}")
            return _resolved_model

    _resolved_model = "gemini-2.0-flash"
    print(f"[RAG] Fallback model: {_resolved_model}")
    return _resolved_model


SYSTEM_INSTRUCTION = """You are 'Solaritivity AI', an expert assistant for a Solar PV Defect Detection System.
You have access to domain knowledge from technical documents and real inspection data
(defect detections, carbon emission analysis) stored in a vector database.

Your role:
- Answer questions about solar panel defects: black core, cracks, finger defects, star cracks, thick lines.
- Provide personalized insights based on actual detection records and carbon emission data.
- When context contains data about a specific image, use that data directly to give precise,
  image-specific answers (defect type, severity, area %, confidence, bounding box, CO2 impact).
- Give actionable maintenance and localized remediation advice based on detected defects.

Guidelines:
- ALWAYS be highly conversational, contextual, crisp, and solution-oriented.
- Limit your INITIAL response to EXACTLY 3 lines of text. Do not give long text walls.
- If asked about a specific image and context is available, focus primarily on that image's data.
- Never fabricate numbers — only use values present in the context.
- At the end of your short 3-line response, ALWAYS ask the user if they would like to know more about the "technical aspects" or "remedy measures".
"""


# ── Image filename extraction ─────────────────────────────────────────────

# Regex patterns to extract image filename from question or history
_IMG_PATTERN = re.compile(
    r"\b([a-f0-9]{8,}\.(?:jpg|jpeg|png|bmp|tiff))\b",
    re.IGNORECASE,
)

def _extract_image_filename(text: str) -> str | None:
    """Extract image filename (UUID-style) from a string."""
    m = _IMG_PATTERN.search(text)
    return m.group(1) if m else None


def _find_image_context(question: str, history: list) -> tuple[str | None, list]:
    """
    Look for an image filename in the question or recent history.
    Returns (image_filename | None, image_specific_docs).
    """
    # Check question first
    img = _extract_image_filename(question)

    # Check last 4 history entries if not found in question
    if not img:
        for h in reversed((history or [])[-4:]):
            img = _extract_image_filename(h.get("content", ""))
            if img:
                break

    if not img:
        return None, []

    # Pull all chunks tagged with that image via filtered cosine search
    docs = query_collection_with_filter(
        query=question,
        where={"image_filename": img},
        n_results=10,
    )
    return img, docs


def _is_image_question(question: str) -> bool:
    """Heuristic: does this question seem to be about a specific uploaded image?"""
    keywords = [
        "this", "the image", "uploaded image", "my image",
        "this panel", "this scan", "current image", "last image",
        "defect in", "defects in", "what was detected", "analysis",
        "result", "severity", "area", "confidence", "co2", "carbon",
        "how many defects", "what defect", "what defects",
        "dominant defect"
    ]
    q_lower = question.lower()
    return any(kw in q_lower for kw in keywords)


# ── Prompt builder ────────────────────────────────────────────────────────

def _build_prompt(question: str, context: str, history: list) -> list:
    """Build Gemini contents array with conversation history."""
    contents = []

    for h in (history or [])[-4:]:
        role = "user" if h.get("role") == "user" else "model"
        contents.append({"role": role, "parts": [{"text": h.get("content", "")}]})

    user_text = (
        f"### Relevant Context from Knowledge Base:\n{context}\n\n"
        f"### Question:\n{question}"
    )
    contents.append({"role": "user", "parts": [{"text": user_text}]})
    return contents


# ── Main entry point ──────────────────────────────────────────────────────

def answer_query(question: str, history: list = None, image_filename: str = None) -> str:
    """
    Generate an answer using RAG + Gemini.

    Parameters
    ----------
    question       : User's question string
    history        : List of {role, content} dicts (conversation history)
    image_filename : Optional — pass the current session's image filename
                     from the frontend to force image-specific context retrieval.
    """
    api_key = _get_api_key()
    model   = _resolve_model(api_key)

    # ── 1. Determine context strategy ─────────────────────────────────────
    image_docs: list = []
    resolved_img      = image_filename

    # Priority 1: explicit filename passed from frontend
    if resolved_img:
        from .vector_store import query_collection_with_filter
        image_docs = query_collection_with_filter(
            query=question,
            where={"image_filename": resolved_img},
            n_results=10,
        )

    # Priority 2: extract filename from question or history text
    if not image_docs:
        resolved_img, image_docs = _find_image_context(question, history or [])

    # Priority 3: if question looks image-related, try latest detection summary
    if not image_docs and _is_image_question(question):
        image_docs = query_collection_with_filter(
            query=question,
            where={"type": "detection_summary"},
            n_results=5,
        )

    # General knowledge retrieval
    general_docs = query_collection(question, n_results=6)

    # ── 2. Assemble context ────────────────────────────────────────────────
    context_parts = []

    if image_docs:
        img_label = f"image '{resolved_img}'" if resolved_img else "the uploaded image"
        context_parts.append(
            f"=== SPECIFIC DATA FOR {img_label.upper()} ===\n" +
            "\n---\n".join(image_docs)
        )

    if general_docs:
        context_parts.append(
            "=== GENERAL KNOWLEDGE BASE ===\n" +
            "\n---\n".join(general_docs)
        )

    context = (
        "\n\n".join(context_parts)
        if context_parts
        else "No relevant context found in knowledge base."
    )

    # ── 3. Build and send request ──────────────────────────────────────────
    contents = _build_prompt(question, context, history or [])
    payload  = {
        "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": contents,
        "generationConfig": {
            "temperature":    float(os.environ.get("LOCAL_LLM_TEMP", "0.7")),
            "maxOutputTokens": int(os.environ.get("LOCAL_LLM_MAX_TOKENS", "512")),
            "topP":           0.95,
        },
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=GEMINI_TIMEOUT_SEC)
            if resp.status_code == 429:
                wait = 2 ** attempt
                print(f"[RAG] Rate limited, retrying in {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            text = (
                data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                    .strip()
            )
            return text if text else "I could not generate a response. Please try again."
        except requests.exceptions.Timeout:
            if attempt == 2:
                raise ValueError("Gemini API timed out. Please try again.")
            time.sleep(2)
        except requests.exceptions.HTTPError as e:
            raise ValueError(
                f"Gemini API error: {e.response.status_code} — {e.response.text[:200]}"
            )
        except Exception as e:
            raise ValueError(f"Unexpected error calling Gemini: {e}")

    raise ValueError("Failed to get response from Gemini after retries.")