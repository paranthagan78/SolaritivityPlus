"""modules/chatbot/rag_engine.py
RAG chatbot using Gemini API — with DistilBERT Sentiment Analysis.

EXISTING FUNCTIONALITY: Fully preserved (image-context RAG, filtered search, etc.)
NEW ADDITIONS:
  - Sentiment analysis on every user message (DistilBERT via sentiment.py)
  - Conversation-level sentiment aggregation (sustained negative mood detection)
  - Empathetic system instruction injected when negative sentiment is detected
  - Empathy knowledge base queried from ChromaDB (support docs in /docs/)
  - Sentiment metadata returned alongside the answer for frontend use
"""
import os
import re
import time
import requests
from dotenv import load_dotenv
load_dotenv()

from .vector_store import query_collection, query_collection_with_filter
from .sentiment import analyze_sentiment, analyze_conversation_sentiment, SentimentResult

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


# ── System instructions ───────────────────────────────────────────────────

BASE_SYSTEM_INSTRUCTION = """You are an expert assistant for a Solar PV Defect Detection System.
You have access to domain knowledge from technical documents and real inspection data
(defect detections, carbon emission analysis) stored in a vector database.

Your role:
- Answer questions about solar panel defects: black core, cracks, finger defects, star cracks, thick lines
- Provide personalized insights based on actual detection records and carbon emission data
- When context contains data about a specific image, use that data directly to give precise,
  image-specific answers (defect type, severity, area %, confidence, bounding box, CO2 impact)
- Give actionable maintenance and remediation advice based on detected defects
- Explain defect severity, causes, and recommended corrective actions
- Help users understand their panel health and carbon footprint

Guidelines:
- Always be concise, factual, and specific
- If context contains exact numbers (confidence, area, CO2), quote them directly
- If asked about a specific image and context is available, focus on that image's data
- If context is insufficient, say so clearly and give general expert guidance
- Never fabricate numbers — only use values present in the context"""

EMPATHY_SYSTEM_EXTENSION = """

────────────────────────────────────────────────────────────────
EMOTIONAL INTELLIGENCE PROTOCOL — ACTIVE
────────────────────────────────────────────────────────────────
The user's message has been detected as expressing NEGATIVE sentiment
(frustration, confusion, worry, or distress). Follow these guidelines:

TONE & APPROACH:
- Lead with acknowledgment of their feeling BEFORE giving technical information
- Use warm, calm, supportive language throughout your response
- Never dismiss or minimize their concern — validate it genuinely
- Avoid overly clinical or blunt phrasing; be human-centered

STRUCTURE (for negative-sentiment responses):
  1. ACKNOWLEDGE  → Briefly name/validate their frustration or concern
                    (e.g., "I completely understand how concerning this must be...")
  2. REASSURE     → Give a grounding statement that you are here to help
  3. EXPLAIN      → Provide the technical answer/information clearly and simply
  4. EMPOWER      → End with a clear, actionable next step they can take
  5. OFFER MORE   → Gently invite further questions

LANGUAGE PATTERNS TO USE:
- "I understand this can feel overwhelming..."
- "That's a completely valid concern..."
- "Let me walk you through this step by step..."
- "You're not alone in finding this confusing..."
- "The good news is that..."
- "Here's exactly what you can do..."
- "Please don't hesitate to ask if anything is unclear..."

LANGUAGE PATTERNS TO AVOID:
- "As I mentioned before..." (implies user wasn't paying attention)
- "Simply do..." / "Just..." (minimizes their difficulty)
- Bullet-only responses without any warm framing
- Abrupt or blunt one-line answers

EMPATHETIC CONTEXT:
If the empathy knowledge base context contains relevant emotional support
guidance (from the support document), weave those strategies naturally
into your response. Do NOT quote the document literally — internalize
and apply its principles.
────────────────────────────────────────────────────────────────"""

SUSTAINED_NEGATIVE_EXTENSION = """

IMPORTANT: This user has shown consistently negative sentiment across
multiple messages in this conversation. They may be increasingly
frustrated or distressed. Be especially patient, warm, and encouraging.
If the situation seems particularly stressful, gently acknowledge that
dealing with equipment issues can be stressful and reassure them that
these problems are solvable with the right approach."""


def _build_system_instruction(sentiment: SentimentResult, is_sustained_negative: bool) -> str:
    """Compose the system instruction based on detected sentiment."""
    instruction = BASE_SYSTEM_INSTRUCTION

    if sentiment.is_negative:
        instruction += EMPATHY_SYSTEM_EXTENSION
        if is_sustained_negative:
            instruction += SUSTAINED_NEGATIVE_EXTENSION

    return instruction


# ── Image filename extraction (UNCHANGED) ─────────────────────────────────

_IMG_PATTERN = re.compile(
    r"\b([a-f0-9]{8,}\.(?:jpg|jpeg|png|bmp|tiff))\b",
    re.IGNORECASE,
)


def _extract_image_filename(text: str) -> str | None:
    m = _IMG_PATTERN.search(text)
    return m.group(1) if m else None


def _find_image_context(question: str, history: list) -> tuple[str | None, list]:
    img = _extract_image_filename(question)
    if not img:
        for h in reversed((history or [])[-4:]):
            img = _extract_image_filename(h.get("content", ""))
            if img:
                break
    if not img:
        return None, []
    docs = query_collection_with_filter(
        query=question,
        where={"image_filename": img},
        n_results=10,
    )
    return img, docs


def _is_image_question(question: str) -> bool:
    keywords = [
        "this image", "the image", "uploaded image", "my image",
        "this panel", "this scan", "current image", "last image",
        "defect in", "defects in", "what was detected", "analysis",
        "result", "severity", "area", "confidence", "co2", "carbon",
        "how many defects", "what defect",
    ]
    q_lower = question.lower()
    return any(kw in q_lower for kw in keywords)


# ── Prompt builder ────────────────────────────────────────────────────────

def _build_prompt(question: str, context: str, history: list) -> list:
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


# ── Empathy context retrieval ─────────────────────────────────────────────

def _get_empathy_context(question: str) -> list:
    """
    Retrieve empathy/support chunks from the vector store.
    These come from the empathetic_support_guide.pdf you place in /docs/.
    Filtered by type='empathy_support'.
    """
    # Try filtered search for empathy docs first
    docs = query_collection_with_filter(
        query=f"emotional support empathy frustration concern {question}",
        where={"type": "empathy_support"},
        n_results=4,
    )
    # Fallback: general search with empathy keywords
    if not docs:
        docs = query_collection(
            query=f"empathetic response emotional support frustrated user {question}",
            n_results=3,
        )
    return docs


# ── Main entry point ──────────────────────────────────────────────────────

def answer_query(
    question: str,
    history:  list = None,
    image_filename: str = None,
) -> dict:
    """
    Generate a sentiment-aware answer using RAG + Gemini + DistilBERT.

    Parameters
    ----------
    question       : User's question string
    history        : List of {role, content} dicts (conversation history)
    image_filename : Optional — current session's image filename for image-specific context

    Returns
    -------
    dict with keys:
        answer          : str   — the generated response text
        sentiment_label : str   — "positive" | "neutral" | "negative"
        sentiment_score : float — confidence 0.0–1.0
        sentiment_compound: float — signed -1.0 to +1.0
        is_negative     : bool  — True if empathetic mode was activated
    """
    api_key = _get_api_key()
    model   = _resolve_model(api_key)

    # ── STEP 1: Sentiment Analysis ─────────────────────────────────────────
    current_sentiment     = analyze_sentiment(question)
    conversation_sentiment = analyze_conversation_sentiment(history or [], window=3)

    # Sustained negative = current is negative AND conversation trend is also negative
    is_sustained_negative = (
        current_sentiment.is_negative and conversation_sentiment.is_negative
    )

    print(
        f"[Sentiment] Current: {current_sentiment} | "
        f"Conversation: {conversation_sentiment} | "
        f"Sustained negative: {is_sustained_negative}"
    )

    # ── STEP 2: Image context (UNCHANGED logic) ───────────────────────────
    image_docs:  list = []
    resolved_img       = image_filename

    if resolved_img:
        image_docs = query_collection_with_filter(
            query=question,
            where={"image_filename": resolved_img},
            n_results=10,
        )

    if not image_docs:
        resolved_img, image_docs = _find_image_context(question, history or [])

    if not image_docs and _is_image_question(question):
        image_docs = query_collection_with_filter(
            query=question,
            where={"type": "detection_summary"},
            n_results=5,
        )

    # ── STEP 3: General + Empathy context ────────────────────────────────
    general_docs = query_collection(question, n_results=6)

    empathy_docs: list = []
    if current_sentiment.is_negative:
        empathy_docs = _get_empathy_context(question)
        print(f"[Sentiment] Empathetic mode ON — {len(empathy_docs)} empathy chunks retrieved")

    # ── STEP 4: Assemble context ──────────────────────────────────────────
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

    if empathy_docs:
        context_parts.append(
            "=== EMPATHETIC SUPPORT GUIDANCE ===\n"
            "(Use these principles to shape your tone and approach — do not quote directly)\n" +
            "\n---\n".join(empathy_docs)
        )

    context = (
        "\n\n".join(context_parts)
        if context_parts
        else "No relevant context found in knowledge base."
    )

    # ── STEP 5: Build system instruction (sentiment-aware) ────────────────
    system_instruction = _build_system_instruction(current_sentiment, is_sustained_negative)

    # ── STEP 6: Build and send Gemini request ─────────────────────────────
    contents = _build_prompt(question, context, history or [])
    payload  = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": contents,
        "generationConfig": {
            "temperature":     float(os.environ.get("LOCAL_LLM_TEMP", "0.7")),
            "maxOutputTokens": int(os.environ.get("LOCAL_LLM_MAX_TOKENS", "2048")),
            "topP":            0.95,
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
            answer = text if text else "I could not generate a response. Please try again."

            # Return answer + sentiment metadata for the frontend
            return {
                "answer":              answer,
                "sentiment_label":     current_sentiment.label,
                "sentiment_score":     current_sentiment.score,
                "sentiment_compound":  current_sentiment.compound,
                "is_negative":         current_sentiment.is_negative,
            }

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