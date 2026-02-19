"""modules/chatbot/ingest.py — Run once to populate ChromaDB."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from modules.chatbot.vector_store import ingest_docs, ingest_csvs

if __name__ == "__main__":
    print("Ingesting documents...")
    ingest_docs()
    print("Ingesting CSV data...")
    ingest_csvs()
    print("Done.")