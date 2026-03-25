"""modules/chatbot/ingest.py — Run once to populate ChromaDB with docs + CSV data."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from modules.chatbot.vector_store import ingest_docs, ingest_csvs, get_stats

if __name__ == "__main__":
    print("=" * 55)
    print("Ingesting PDF documents from docs/ folder...")
    doc_count = ingest_docs()
    print(f"  → {doc_count} document chunks ingested")

    print("\nIngesting CSV data...")
    print("  detections.csv + carbon.csv")
    csv_count = ingest_csvs()
    print(f"  → {csv_count} CSV rows ingested")

    print("\nVector store stats:")
    print(" ", get_stats())
    print("=" * 55)
    print("Done. ChromaDB is ready.")
    print()
    print("NOTE: Live image results are ingested automatically")
    print("via POST /api/chat/ingest_image after each detection run.")