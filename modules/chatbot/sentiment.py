"""modules/chatbot/sentiment.py  [FIXED v3 - eager model warmup]
Transformer-based sentiment analysis using DistilBERT.

Model: distilbert-base-uncased-finetuned-sst-2-english

FIXES IN v3:
  - Added warmup_sentiment_model() for eager loading at app startup.
    Previously the lazy-load design meant teammates who hadn't downloaded
    the model would silently fall back to rule-based sentiment for ALL
    requests, since the model hadn't loaded before requests started arriving.
  - Added _model_loaded / _model_failed flags so load state is always known.
  - Warm-up inference pass runs after load so first real request isn't slow.

GUARDS AGAINST DISTILBERT FALSE POSITIVES (from v2, preserved):
  1. QUESTION GUARD      — plain informational questions forced to neutral
  2. RAISED THRESHOLDS   — neutral zone 0.80, negative trigger -0.55
  3. EMOTION KEYWORD GATE — must contain a real emotional word to trigger
                            empathetic mode (prevents "what defects are in
                            my image?" from triggering empathy)
"""

from __future__ import annotations
import re
from dataclasses import dataclass


# ── Result dataclass ──────────────────────────────────────────────────────

@dataclass
class SentimentResult:
    label: str          # "positive" | "neutral" | "negative"
    score: float        # 0.0-1.0 confidence for the label
    is_negative: bool   # True when empathetic mode should activate
    compound: float     # -1.0 (very negative) to +1.0 (very positive)

    def __repr__(self):
        sign = "+" if self.compound >= 0 else ""
        return (
            f"SentimentResult(label={self.label!r}, "
            f"score={self.score:.3f}, compound={sign}{self.compound:.3f})"
        )


# ── Tuning constants ──────────────────────────────────────────────────────

NEUTRAL_THRESHOLD   = 0.80   # DistilBERT confidence below this -> "neutral"
NEGATIVE_TRIGGER    = -0.55  # compound must be this negative to trigger empathy
MIN_WORDS_FOR_MODEL = 6      # messages shorter than this skip the model

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


# ── Model state flags (module-level) ─────────────────────────────────────
# These are checked by app.py health endpoint and _get_pipeline()

_pipeline     = None   # the loaded HuggingFace pipeline object
_model_loaded = False  # True once model confirmed ready
_model_failed = False  # True if transformers not installed or load failed


# ── Guard 1: Question / informational intent detection ────────────────────
# Catches "what defects are present", "show me results", "how many cracks" etc.

_QUESTION_STARTERS = re.compile(
    r"^(what|which|where|when|who|how|why|is|are|was|were|do|does|did|can|could|"
    r"should|would|will|tell me|show me|list|give me|explain|describe|find|get|"
    r"display|print|calculate|analyse|analyze|check|detect|identify|summarize|"
    r"summarise|compare|help me|can you|could you|please)['\s,]",
    re.IGNORECASE,
)

_ENDS_WITH_QUESTION = re.compile(r"\?\s*$")

_FACTUAL_PHRASES = re.compile(
    r"\b(defect|crack|panel|image|result|detect|analysis|carbon|co2|severity|"
    r"confidence|area|bounding|bbox|upload|scan|percentage|present|found|show|"
    r"list|how many|what type|which|status|report|output|data|file|model|"
    r"infrared|thermal|el image|star.crack|black.core|finger|thick.line)\b",
    re.IGNORECASE,
)


def _is_informational_question(text: str) -> bool:
    """
    Return True if the message is a plain informational question or command.
    These are ALWAYS forced to neutral regardless of DistilBERT's output.
    """
    stripped   = text.strip()
    word_count = len(stripped.split())
    if _QUESTION_STARTERS.match(stripped):
        return True
    if _ENDS_WITH_QUESTION.search(stripped):
        return True
    if word_count < MIN_WORDS_FOR_MODEL and _FACTUAL_PHRASES.search(stripped):
        return True
    return False


# ── Guard 3: Genuine emotion keyword requirement ──────────────────────────

_EMOTION_KEYWORDS = re.compile(
    r"\b(frustrated|frustrating|frustration|angry|upset|worried|worry|anxious|"
    r"stressed|stress|overwhelmed|confused|confusing|lost|helpless|terrible|"
    r"horrible|awful|worst|useless|wrong|disappointed|disappointing|"
    r"annoyed|annoying|unhappy|failed|failure|crash|stuck|"
    r"don.?t understand|can.?t figure|cannot figure|no idea|nothing works|"
    r"doesn.?t work|not working|not helpful|still broken|"
    r"give up|hopeless|disaster|struggling|struggle|difficult|hard time|"
    r"scared|afraid|fear|hurt|pain|sad|miserable|distressed|not sure|"
    r"uncertain|why isn.?t|why is this|this is wrong|this is bad)\b",
    re.IGNORECASE,
)


def _has_emotion_signal(text: str) -> bool:
    """Return True if text contains at least one genuine emotional keyword."""
    return bool(_EMOTION_KEYWORDS.search(text))


# ── Eager warmup (call this from app.py at startup) ───────────────────────

def warmup_sentiment_model() -> bool:
    """
    Eagerly load and warm up the DistilBERT model.

    Call this ONCE at application startup inside create_app() in app.py.
    Blocks until the model is downloaded (~67 MB, first run only) and loaded.

    This prevents the teammate problem: without eager loading, the lazy-load
    design means the first N requests all arrive before the model is ready
    and silently fall back to rule-based sentiment instead of DistilBERT.

    Returns
    -------
    bool — True if DistilBERT loaded successfully,
           False if falling back to rule-based (transformers not installed).
    """
    global _pipeline, _model_loaded, _model_failed

    if _model_loaded:
        print("[Sentiment] Model already loaded, skipping warmup.")
        return True
    if _model_failed:
        print("[Sentiment] Model previously failed to load, skipping warmup.")
        return False

    print("[Sentiment] ── Warming up DistilBERT sentiment model ──────────")
    print(f"[Sentiment] Model : {MODEL_NAME}")
    print("[Sentiment] Note  : First run downloads ~67 MB (cached after that)...")

    try:
        from transformers import pipeline as hf_pipeline

        _pipeline = hf_pipeline(
            task="sentiment-analysis",
            model=MODEL_NAME,
        )

        # Warm-up inference pass — compiles the model graph so the first
        # real user request isn't slower than subsequent ones
        _ = _pipeline("Solar panel inspection warm-up pass.")

        _model_loaded = True
        print("[Sentiment] ✓ DistilBERT loaded and ready.")
        return True

    except ImportError:
        _model_failed = True
        print(
            "[Sentiment] ✗ 'transformers' package not found.\n"
            "  Fix: pip install transformers torch\n"
            "  Falling back to rule-based sentiment analysis."
        )
        return False

    except Exception as e:
        _model_failed = True
        print(
            f"[Sentiment] ✗ Model load failed: {e}\n"
            f"  Falling back to rule-based sentiment analysis."
        )
        return False


def _get_pipeline():
    """
    Return the loaded pipeline.

    Normally _pipeline is already set by warmup_sentiment_model() at startup.
    This function is a safety net in case warmup was somehow skipped —
    it triggers a lazy load rather than crashing. But if app.py is correct,
    this path should never be needed.
    """
    global _model_loaded, _model_failed

    if _model_loaded:
        return _pipeline
    if _model_failed:
        return None

    # Safety net: warmup wasn't called at startup — trigger it now
    print(
        "[Sentiment] WARNING: warmup_sentiment_model() was not called at startup.\n"
        "  Add it to create_app() in app.py for reliable behaviour.\n"
        "  Triggering lazy load now..."
    )
    warmup_sentiment_model()
    return _pipeline


# ── Rule-based fallback ───────────────────────────────────────────────────
# Used ONLY when transformers is not installed or model load failed.

_NEG_WORDS = {
    "terrible", "horrible", "awful", "worst", "useless", "frustrated",
    "frustrating", "angry", "upset", "confused", "error", "failed",
    "failure", "wrong", "disappointed", "annoyed", "problem", "crash",
    "stuck", "worried", "anxious", "stressed", "overwhelmed", "helpless",
    "don't understand", "cant figure", "nothing works", "not working",
    "not helpful",
}

_POS_WORDS = {
    "great", "good", "excellent", "perfect", "awesome", "wonderful",
    "thank", "thanks", "helpful", "clear", "understood", "working",
    "fixed", "solved", "resolved", "appreciate", "love", "amazing",
    "nice", "correct", "right", "easy", "simple",
}

_INTENSIFIERS = {"very", "really", "extremely", "so", "quite", "absolutely"}


def _rule_based_sentiment(text: str) -> SentimentResult:
    words   = re.findall(r"\b\w+\b", text.lower())
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]

    neg_hits = sum(1.5 if words[max(0, i-1)] in _INTENSIFIERS else 1.0
                   for i, w in enumerate(words) if w in _NEG_WORDS)
    pos_hits = sum(1.5 if words[max(0, i-1)] in _INTENSIFIERS else 1.0
                   for i, w in enumerate(words) if w in _POS_WORDS)
    neg_hits += sum(1.0 for bg in bigrams if bg in _NEG_WORDS)
    pos_hits += sum(1.0 for bg in bigrams if bg in _POS_WORDS)

    total    = neg_hits + pos_hits + 1e-9
    compound = (pos_hits - neg_hits) / max(total, 1)
    compound = max(-1.0, min(1.0, compound))

    if compound < -0.15:
        label = "negative"
        score = min(0.5 + abs(compound) * 0.5, 0.99)
    elif compound > 0.15:
        label = "positive"
        score = min(0.5 + compound * 0.5, 0.99)
    else:
        label = "neutral"
        score = 0.60

    is_neg = (compound <= NEGATIVE_TRIGGER) and _has_emotion_signal(text)
    return SentimentResult(label=label, score=score, is_negative=is_neg, compound=compound)


# ── Public API ────────────────────────────────────────────────────────────

def analyze_sentiment(text: str) -> SentimentResult:
    """
    Analyze the sentiment of a user message using DistilBERT.

    Parameters
    ----------
    text : str  — raw user message string

    Returns
    -------
    SentimentResult — label, score, is_negative, compound
    """
    if not text or not text.strip():
        return SentimentResult(label="neutral", score=1.0, is_negative=False, compound=0.0)

    # Guard 1: plain informational questions -> always neutral
    if _is_informational_question(text):
        print("[Sentiment] Informational question -> forced neutral")
        return SentimentResult(label="neutral", score=0.95, is_negative=False, compound=0.0)

    # Guard 2: too short for reliable DistilBERT inference
    word_count = len(text.split())
    if word_count < MIN_WORDS_FOR_MODEL:
        print(f"[Sentiment] Short message ({word_count} words) -> forced neutral")
        return SentimentResult(label="neutral", score=0.90, is_negative=False, compound=0.0)

    # Truncate to 512 words for DistilBERT context window
    # NOTE: this only affects the sentiment INPUT — has zero effect on chatbot output length
    truncated = " ".join(text.split()[:512])
    pipe = _get_pipeline()

    if pipe is not None:
        try:
            result    = pipe(truncated)[0]
            raw_label = result["label"].lower()   # "positive" or "negative"
            raw_score = float(result["score"])    # 0.0–1.0

            compound = raw_score if raw_label == "positive" else -raw_score

            # Carve out neutral zone for low-confidence predictions
            if raw_score < NEUTRAL_THRESHOLD:
                label    = "neutral"
                compound = compound * (raw_score / NEUTRAL_THRESHOLD)
            else:
                label = raw_label

            # Guard 3: require real emotion word for negative trigger
            is_neg = (
                label == "negative"
                and compound <= NEGATIVE_TRIGGER
                and _has_emotion_signal(text)
            )

            return SentimentResult(
                label=label,
                score=raw_score,
                is_negative=is_neg,
                compound=round(compound, 4),
            )

        except Exception as e:
            print(f"[Sentiment] DistilBERT inference error: {e}. Using fallback.")

    return _rule_based_sentiment(text)


def analyze_conversation_sentiment(history: list, window: int = 3) -> SentimentResult:
    """
    Aggregate sentiment across the last `window` user messages.
    More recent messages are weighted more heavily (recency weighting).

    Parameters
    ----------
    history : list[dict]  — each dict has keys "role" and "content"
    window  : int         — how many recent user messages to consider

    Returns
    -------
    SentimentResult — weighted aggregate result
    """
    user_msgs = [
        h["content"] for h in history
        if h.get("role") == "user" and h.get("content", "").strip()
    ][-window:]

    if not user_msgs:
        return SentimentResult(label="neutral", score=1.0, is_negative=False, compound=0.0)

    weights  = list(range(1, len(user_msgs) + 1))
    results  = [analyze_sentiment(msg) for msg in user_msgs]
    total_w  = sum(weights)

    compound = sum(r.compound * w for r, w in zip(results, weights)) / total_w
    compound = round(max(-1.0, min(1.0, compound)), 4)

    label_scores: dict = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    for r, w in zip(results, weights):
        label_scores[r.label] += w
    label = max(label_scores, key=label_scores.get)

    score = sum(r.score * w for r, w in zip(results, weights)) / total_w

    # Sustained negative also requires at least one genuine emotional message
    any_emotion = any(_has_emotion_signal(m) for m in user_msgs)
    is_neg = (compound <= NEGATIVE_TRIGGER) and any_emotion

    return SentimentResult(
        label=label,
        score=round(score, 4),
        is_negative=is_neg,
        compound=compound,
    )