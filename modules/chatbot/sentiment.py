"""modules/chatbot/sentiment.py
Transformer-based sentiment analysis using DistilBERT.

Model: distilbert-base-uncased-finetuned-sst-2-english

ROOT CAUSE OF THE BUG (and fixes applied):
  DistilBERT was trained on the SST-2 movie review dataset.
  It treats domain words like "defect", "crack", "broken", "damage" as
  strongly negative — because in movie reviews, those words ARE negative.
  In a solar inspection chatbot, they are neutral technical terms.

  Three-layer defence added:
    1. QUESTION GUARD     — plain informational questions (what/how/where/
                            show/list/detect...) are forced to neutral BEFORE
                            reaching the model. "What defects are in this image?"
                            is a query, not a complaint.
    2. RAISED THRESHOLDS  — neutral zone raised from 0.65 → 0.80 confidence.
                            Negative trigger tightened from -0.20 → -0.55.
    3. EMOTION KEYWORD GATE — even if DistilBERT says "negative" with high
                            confidence, empathetic mode only activates when
                            the message contains a genuine emotional word
                            (frustrated, worried, confused, etc.).
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

# DistilBERT confidence below this -> "neutral" (conservative to avoid false negatives)
NEUTRAL_THRESHOLD = 0.80

# Compound must be THIS negative to trigger empathetic mode
NEGATIVE_TRIGGER  = -0.55

# Messages shorter than this skip the model (unreliable on very short text)
MIN_WORDS_FOR_MODEL = 6


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

    # Short message containing only technical vocabulary -> query, not emotion
    if word_count < MIN_WORDS_FOR_MODEL and _FACTUAL_PHRASES.search(stripped):
        return True

    return False


# ── Guard 3: Genuine emotion keyword requirement ──────────────────────────
# DistilBERT can fire "negative" on purely technical text.
# We require at least one real emotional word to be present before
# activating empathetic mode.

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


# ── DistilBERT model (lazy-loaded singleton) ──────────────────────────────

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    try:
        from transformers import pipeline as hf_pipeline
        print("[Sentiment] Loading DistilBERT sentiment model...")
        _pipeline = hf_pipeline(
            task="sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
        print("[Sentiment] DistilBERT model loaded successfully.")
    except ImportError:
        print(
            "[Sentiment] WARNING: 'transformers' not installed. "
            "Run: pip install transformers torch\n"
            "Falling back to rule-based sentiment."
        )
        _pipeline = None
    return _pipeline


# ── Rule-based fallback ───────────────────────────────────────────────────

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
    Analyze the sentiment of a user message.

    Parameters
    ----------
    text : str  - the raw user message string.

    Returns
    -------
    SentimentResult - label, score, is_negative, compound
    """
    if not text or not text.strip():
        return SentimentResult(label="neutral", score=1.0, is_negative=False, compound=0.0)

    # Guard 1: plain informational questions -> always neutral
    if _is_informational_question(text):
        print(f"[Sentiment] Informational question detected -> forced neutral")
        return SentimentResult(label="neutral", score=0.95, is_negative=False, compound=0.0)

    # Guard 2: too short for reliable DistilBERT inference
    word_count = len(text.split())
    if word_count < MIN_WORDS_FOR_MODEL:
        print(f"[Sentiment] Short message ({word_count} words) -> forced neutral")
        return SentimentResult(label="neutral", score=0.90, is_negative=False, compound=0.0)

    truncated = " ".join(text.split()[:512])
    pipe = _get_pipeline()

    if pipe is not None:
        try:
            result    = pipe(truncated)[0]
            raw_label = result["label"].lower()
            raw_score = float(result["score"])

            compound = raw_score if raw_label == "positive" else -raw_score

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
    More recent messages are weighted more heavily.
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

    # Sustained negative also requires at least one emotional message
    any_emotion = any(_has_emotion_signal(m) for m in user_msgs)
    is_neg = (compound <= NEGATIVE_TRIGGER) and any_emotion

    return SentimentResult(
        label=label,
        score=round(score, 4),
        is_negative=is_neg,
        compound=compound,
    )