"""modules/chatbot/vector_store.py
ChromaDB vector store with sentence-transformers embeddings (fully local).
Ingests PDF docs + detection/carbon CSV data + per-image results for personalized RAG.
Uses cosine similarity for all searches.
"""
import os
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from config import CHROMA_FOLDER, DETECTIONS_CSV, CARBON_CSV, DOCS_FOLDER

COLLECTION_NAME = "solar_pv_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_client     = None
_collection = None

# ── Severity + action mapping ─────────────────────────────────────────────
DEFECT_SEVERITY = {
    "black_core":  "Critical",
    "crack":       "High",
    "star_crack":  "High",
    "finger":      "Medium",
    "thick_line":  "Low",
}

DEFECT_ACTIONS = {
    "black_core":  "Immediate panel replacement recommended. Black core indicates severe cell failure.",
    "crack":       "Schedule urgent inspection. Cracks propagate and cause significant power loss.",
    "star_crack":  "Schedule urgent inspection. Star cracks indicate mechanical stress damage.",
    "finger":      "Monitor closely. Finger defects affect current collection efficiency.",
    "thick_line":  "Schedule routine maintenance. Thick lines may indicate metallization issues.",
}


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
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def query_collection(query: str, n_results: int = 6) -> list:
    """Cosine similarity search — returns top-n matching document strings."""
    try:
        col   = get_collection()
        count = col.count()
        n     = min(n_results, count) if count > 0 else 0
        if n == 0:
            return []
        results = col.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        return results.get("documents", [[]])[0]
    except Exception as e:
        print(f"[VectorStore] Query error: {e}")
        return []


def query_collection_with_filter(query: str, where: dict, n_results: int = 10) -> list:
    """Cosine similarity search filtered by metadata (e.g. image_filename or type)."""
    try:
        col   = get_collection()
        count = col.count()
        n     = min(n_results, count) if count > 0 else 0
        if n == 0:
            return []
        results = col.query(
            query_texts=[query],
            n_results=n,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return results.get("documents", [[]])[0]
    except Exception as e:
        print(f"[VectorStore] Filtered query error: {e}")
        return []


# ── Document ingestion ────────────────────────────────────────────────────

def ingest_docs():
    """Ingest PDF documents from DOCS_FOLDER into ChromaDB."""
    from pypdf import PdfReader
    col   = get_collection()
    count = 0
    if not os.path.isdir(DOCS_FOLDER):
        print(f"[RAG] DOCS_FOLDER not found: {DOCS_FOLDER}")
        return count

    for fname in os.listdir(DOCS_FOLDER):
        if not fname.lower().endswith(".pdf"):
            continue
        fpath = os.path.join(DOCS_FOLDER, fname)
        try:
            reader = PdfReader(fpath)
            ids, texts, metas = [], [], []
            for i, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                for j in range(0, len(text), 800):
                    chunk = text[j:j + 800].strip()
                    if len(chunk) > 50:
                        ids.append(f"{fname}_p{i}_c{j}")
                        texts.append(chunk)
                        metas.append({
                            "source":         fname,
                            "page":           i,
                            "type":           "document",
                            "image_filename": "",
                        })
            if ids:
                col.upsert(ids=ids, documents=texts, metadatas=metas)
                count += len(ids)
                print(f"[RAG] {fname}: {len(ids)} chunks ingested")
        except Exception as e:
            print(f"[RAG] Error ingesting {fname}: {e}")
    return count


# ── CSV formatters ────────────────────────────────────────────────────────

def _format_detection_row(row: pd.Series) -> str:
    defect   = str(row.get("defect_class", "unknown"))
    severity = DEFECT_SEVERITY.get(defect, "Unknown")
    action   = DEFECT_ACTIONS.get(defect, "Consult a technician.")

    parts = []
    if "timestamp"      in row.index: parts.append(f"Date: {row['timestamp']}")
    if "image_filename" in row.index: parts.append(f"Image: {row['image_filename']}")
    parts.append(f"Defect detected: {defect}")
    parts.append(f"Severity: {severity}")
    if "confidence"  in row.index: parts.append(f"Confidence: {float(row['confidence']):.2%}")
    if "area_ratio"  in row.index: parts.append(f"Defect area: {float(row['area_ratio'])*100:.2f}% of panel")
    if "bbox_x1"     in row.index:
        parts.append(
            f"Bounding box: ({row['bbox_x1']},{row['bbox_y1']}) → ({row['bbox_x2']},{row['bbox_y2']})"
        )
    if "image_width" in row.index and "image_height" in row.index:
        parts.append(f"Image size: {row['image_width']}x{row['image_height']}px")
    parts.append(f"Recommended action: {action}")
    return " | ".join(parts)


def _format_carbon_row(row: pd.Series) -> str:
    parts = []
    if "timestamp"          in row.index: parts.append(f"Date: {row['timestamp']}")
    if "image_filename"     in row.index: parts.append(f"Image: {row['image_filename']}")
    if "city"               in row.index: parts.append(f"City: {row['city']}")
    if "panel_power_w"      in row.index: parts.append(f"Panel power: {row['panel_power_w']}W")
    if "ambient_temp_c"     in row.index: parts.append(f"Ambient temperature: {row['ambient_temp_c']}°C")
    if "irradiance_w_m2"    in row.index: parts.append(f"Irradiance: {row['irradiance_w_m2']} W/m²")
    if "emission_factor"    in row.index: parts.append(f"Emission factor: {row['emission_factor']}")
    if "num_defects"        in row.index: parts.append(f"Number of defects: {row['num_defects']}")
    if "dominant_defect"    in row.index:
        dom = str(row['dominant_defect'])
        sev = DEFECT_SEVERITY.get(dom, "Unknown")
        parts.append(f"Dominant defect: {dom} (Severity: {sev})")
    if "total_degradation_pct" in row.index: parts.append(f"Total degradation: {row['total_degradation_pct']}%")
    if "co2_kg_per_year"    in row.index: parts.append(f"CO2 emission per year: {row['co2_kg_per_year']} kg")
    if not parts:
        parts = [f"{k}: {v}" for k, v in row.items() if pd.notna(v)]
    return " | ".join(parts)


def ingest_csvs():
    """Ingest detection and carbon CSV rows into ChromaDB."""
    col   = get_collection()
    total = 0

    csv_configs = [
        (DETECTIONS_CSV, "det",    "detection", _format_detection_row),
        (CARBON_CSV,     "carbon", "carbon",    _format_carbon_row),
    ]

    for csv_path, prefix, doc_type, formatter in csv_configs:
        if not os.path.isfile(csv_path):
            print(f"[RAG] CSV not found, skipping: {csv_path}")
            continue
        try:
            df = pd.read_csv(csv_path)
            ids, texts, metas = [], [], []
            for i, row in df.iterrows():
                text = formatter(row)
                if not text.strip():
                    continue
                img_fname = str(row.get("image_filename", ""))
                ids.append(f"{prefix}_row_{i}")
                texts.append(text)
                metas.append({
                    "source":         os.path.basename(csv_path),
                    "type":           doc_type,
                    "row":            int(i),
                    "image_filename": img_fname,
                })

            if ids:
                for bs in range(0, len(ids), 200):
                    col.upsert(
                        ids=ids[bs:bs+200],
                        documents=texts[bs:bs+200],
                        metadatas=metas[bs:bs+200],
                    )
                total += len(ids)
                print(f"[RAG] {os.path.basename(csv_path)}: {len(ids)} rows ingested")
        except Exception as e:
            print(f"[RAG] CSV error ({csv_path}): {e}")

    return total


# ── Live per-image ingestion ──────────────────────────────────────────────

def ingest_image_result(image_filename: str, detections: list, carbon_data: dict = None):
    """
    Ingest a single image's detection + carbon results into ChromaDB
    immediately after the analysis pipeline runs.

    Call this from modules/detection and modules/carbon after each upload.

    Parameters
    ----------
    image_filename : str
        Filename of the uploaded image, e.g. "abc123.jpg"
    detections : list[dict]
        Keys per item: defect_class, confidence, area_ratio,
                       bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                       image_width, image_height
    carbon_data : dict | None
        Keys: city, panel_power_w, ambient_temp_c, irradiance_w_m2,
              emission_factor, num_defects, dominant_defect,
              total_degradation_pct, co2_kg_per_year
    """
    col = get_collection()
    ids, texts, metas = [], [], []

    # ── Individual detection chunks ───────────────────────────────────────
    for i, det in enumerate(detections):
        defect   = str(det.get("defect_class", "unknown"))
        severity = DEFECT_SEVERITY.get(defect, "Unknown")
        action   = DEFECT_ACTIONS.get(defect, "Consult a technician.")
        area_pct = float(det.get("area_ratio", 0)) * 100
        conf     = float(det.get("confidence", 0))

        text = (
            f"Image: {image_filename} | "
            f"Defect detected: {defect} | "
            f"Severity: {severity} | "
            f"Confidence: {conf:.2%} | "
            f"Defect area: {area_pct:.2f}% of panel | "
            f"Bounding box: ({det.get('bbox_x1')},{det.get('bbox_y1')}) → "
            f"({det.get('bbox_x2')},{det.get('bbox_y2')}) | "
            f"Image size: {det.get('image_width')}x{det.get('image_height')}px | "
            f"Recommended action: {action}"
        )
        ids.append(f"img_{image_filename}_det_{i}")
        texts.append(text)
        metas.append({
            "source":         "live_detection",
            "type":           "detection",
            "image_filename": image_filename,
        })

    # ── Summary chunk (aggregate over all defects) ────────────────────────
    if detections:
        defect_counts: dict = {}
        for d in detections:
            k = d.get("defect_class", "unknown")
            defect_counts[k] = defect_counts.get(k, 0) + 1

        total_area  = sum(float(d.get("area_ratio", 0)) * 100 for d in detections)
        severities  = [DEFECT_SEVERITY.get(d.get("defect_class", ""), "Low") for d in detections]
        top_sev     = (
            "Critical" if "Critical" in severities else
            "High"     if "High"     in severities else
            "Medium"   if "Medium"   in severities else "Low"
        )
        breakdown   = ", ".join(f"{k} x{v}" for k, v in defect_counts.items())

        summary = (
            f"Image: {image_filename} | "
            f"Total defects found: {len(detections)} | "
            f"Defect breakdown: {breakdown} | "
            f"Total defect area coverage: {total_area:.2f}% | "
            f"Overall severity: {top_sev}"
        )
        ids.append(f"img_{image_filename}_summary")
        texts.append(summary)
        metas.append({
            "source":         "live_detection",
            "type":           "detection_summary",
            "image_filename": image_filename,
        })

    # ── Carbon chunk ──────────────────────────────────────────────────────
    if carbon_data:
        dom   = str(carbon_data.get("dominant_defect", "unknown"))
        sev   = DEFECT_SEVERITY.get(dom, "Unknown")
        c_text = (
            f"Image: {image_filename} | "
            f"City: {carbon_data.get('city', 'N/A')} | "
            f"Panel power: {carbon_data.get('panel_power_w', 'N/A')}W | "
            f"Ambient temperature: {carbon_data.get('ambient_temp_c', 'N/A')}°C | "
            f"Irradiance: {carbon_data.get('irradiance_w_m2', 'N/A')} W/m² | "
            f"Number of defects: {carbon_data.get('num_defects', 'N/A')} | "
            f"Dominant defect: {dom} (Severity: {sev}) | "
            f"Total degradation: {carbon_data.get('total_degradation_pct', 'N/A')}% | "
            f"CO2 emission per year: {carbon_data.get('co2_kg_per_year', 'N/A')} kg"
        )
        ids.append(f"img_{image_filename}_carbon")
        texts.append(c_text)
        metas.append({
            "source":         "live_carbon",
            "type":           "carbon",
            "image_filename": image_filename,
        })

    if ids:
        col.upsert(ids=ids, documents=texts, metadatas=metas)
        print(f"[RAG] Ingested {len(ids)} chunks for image: {image_filename}")


def get_stats() -> dict:
    """Return collection stats."""
    try:
        col = get_collection()
        return {"total_documents": col.count(), "collection": COLLECTION_NAME}
    except Exception as e:
        return {"error": str(e)}