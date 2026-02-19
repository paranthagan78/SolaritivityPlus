"""modules/chatbot/vector_store.py
ChromaDB vector store with sentence-transformers embeddings (fully local).
"""
import os
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from config import CHROMA_FOLDER, DETECTIONS_CSV, CARBON_CSV, DOCS_FOLDER

COLLECTION_NAME = "solar_pv_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # ~80 MB, downloads once

_client     = None
_collection = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_FOLDER)
    return _client


def get_collection():
    global _collection
    if _collection is not None:
        return _collection
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    _collection = _get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )
    return _collection


def query_collection(query: str, n_results: int = 5) -> list:
    try:
        results = get_collection().query(query_texts=[query], n_results=n_results)
        return results.get("documents", [[]])[0]
    except Exception as e:
        print(f"[VectorStore] Query error: {e}")
        return []


def ingest_docs():
    from pypdf import PdfReader
    col = get_collection()
    count = 0
    for fname in os.listdir(DOCS_FOLDER):
        if not fname.lower().endswith(".pdf"):
            continue
        try:
            reader = PdfReader(os.path.join(DOCS_FOLDER, fname))
            ids, texts = [], []
            for i, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                for j in range(0, len(text), 800):
                    chunk = text[j:j + 800].strip()
                    if chunk:
                        ids.append(f"{fname}_p{i}_c{j}")
                        texts.append(chunk)
            if ids:
                col.upsert(ids=ids, documents=texts,
                           metadatas=[{"source": fname}] * len(ids))
                count += len(ids)
                print(f"[RAG] {fname}: {len(ids)} chunks ingested")
        except Exception as e:
            print(f"[RAG] Error ingesting {fname}: {e}")
    return count


def ingest_csvs():
    col   = get_collection()
    total = 0
    for csv_path, prefix in [(DETECTIONS_CSV, "det"), (CARBON_CSV, "carbon")]:
        if not os.path.isfile(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
            ids, texts, metas = [], [], []
            for i, row in df.iterrows():
                ids.append(f"{prefix}_row_{i}")
                texts.append(" | ".join(f"{k}={v}" for k, v in row.items()))
                metas.append({"source": prefix})
            if ids:
                col.upsert(ids=ids, documents=texts, metadatas=metas)
                total += len(ids)
                print(f"[RAG] {os.path.basename(csv_path)}: {len(ids)} rows ingested")
        except Exception as e:
            print(f"[RAG] CSV error ({csv_path}): {e}")
    return total