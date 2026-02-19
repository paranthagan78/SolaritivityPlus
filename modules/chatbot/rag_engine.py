"""modules/chatbot/rag_engine.py
RAG chatbot using local GGUF model (orca-mini-3b) via ctransformers.
Retrieves context from ChromaDB, then generates answer locally.
"""
import re
from ctransformers import AutoModelForCausalLM
from config import LOCAL_LLM_PATH, LOCAL_LLM_TYPE, LOCAL_LLM_MAX_TOKENS, LOCAL_LLM_TEMP
from .vector_store import query_collection

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        print(f"[RAG] Loading local model from: {LOCAL_LLM_PATH}")
        _llm = AutoModelForCausalLM.from_pretrained(
            LOCAL_LLM_PATH,
            model_type=LOCAL_LLM_TYPE,
            max_new_tokens=LOCAL_LLM_MAX_TOKENS,
            temperature=LOCAL_LLM_TEMP,
            local_files_only=True,
        )
        print("[RAG] Model loaded.")
    return _llm


def _preprocess(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9.,!?%:/()\-\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _build_prompt(question: str, context: str, history: list) -> str:
    system = (
        "You are a solar PV defect detection expert assistant. "
        "Use the provided context from domain documents and inspection CSV data "
        "to answer questions accurately and concisely. "
        "If the context is insufficient, say so clearly."
    )
    history_text = ""
    if history:
        for h in history[-4:]:   # keep last 4 turns to stay within context window
            role  = "User"  if h.get("role") == "user"  else "Assistant"
            history_text += f"{role}: {_preprocess(h.get('content',''))}\n"

    prompt = (
        f"### System:\n{system}\n\n"
        f"### Context:\n{_preprocess(context)}\n\n"
    )
    if history_text:
        prompt += f"### Conversation History:\n{history_text}\n"

    prompt += f"### User:\n{_preprocess(question)}\n\n### Response:\n"
    return prompt


def answer_query(question: str, history: list = None) -> str:
    # 1. Retrieve relevant docs from ChromaDB
    docs    = query_collection(question, n_results=5)
    context = "\n---\n".join(docs) if docs else "No relevant context found."

    # 2. Build prompt
    prompt = _build_prompt(question, context, history or [])

    # 3. Run local model (streaming, collect all tokens)
    llm      = _get_llm()
    response = ""
    for token in llm(prompt, stream=True):
        response += token

    return response.strip()