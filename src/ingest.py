"""
ingest.py — one-time (or whenever the dataset changes) script that turns the
curated respite-service records into a searchable vector index.

Run from the repo root:  python src/ingest.py
Rebuild whenever data/services.json changes, or after switching EMBEDDING_BACKEND.
"""

import json  # for loading the curated service records from disk
import os    # for reading which embedding backend to use from the environment
from pathlib import Path  # for building filesystem paths that do not depend on the caller's working directory

from dotenv import load_dotenv  # pulls EMBEDDING_BACKEND / AWS_* out of the local .env so dev and demo share one config surface
from langchain_community.vectorstores import FAISS  # lightweight, zero-infra vector store — good fit for a few dozen/hundred records

load_dotenv()  # load .env once at import time so os.getenv() below sees the project's configured backend

# Anchor every data path to the repo root (this file lives in src/), so `python src/ingest.py`
# and `pytest` from anywhere both resolve data/ to the same place.
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../hackathon
DATA_DIR = PROJECT_ROOT / "data"                    # .../hackathon/data
DEFAULT_SERVICES_PATH = DATA_DIR / "services.json"  # the real curated dataset (copy of services.sample.json to start)
DEFAULT_INDEX_PATH = DATA_DIR / "vector_index"      # where the built FAISS index is persisted


def load_services(path: Path | str = DEFAULT_SERVICES_PATH) -> list[dict]:
    """Load the curated respite-service records (JSON list of dicts) from disk."""
    with open(path, "r", encoding="utf-8") as f:   # open the dataset file for reading
        return json.load(f)                         # parse and return the list of service records


def to_document_text(service: dict) -> str:
    """
    Flatten one structured service record into a single natural-language
    passage. This is the text that actually gets embedded, so it should read
    the way a caregiver's need would be phrased against it.
    """
    return (
        f"{service['name']} is a {service['care_type']} service in {service['area']}. "   # name + type + location up front
        f"Open {service['days']}, {service['hours']}. "                                     # schedule, phrased in plain language
        f"Costs approximately S${service['cost_per_session']} per session. "                # cost, kept explicit for transparency
        f"{service.get('description', '')}"                                                  # the rich free-text description carries the real semantic signal
    )


def get_embeddings():
    """
    Return the embedding backend to use, switchable via an environment
    variable so local/free development and the Bedrock demo build share the
    same ingestion script.
    """
    backend = os.getenv("EMBEDDING_BACKEND", "local")  # default to free local embeddings for day-to-day dev

    if backend == "bedrock":
        from langchain_community.embeddings import BedrockEmbeddings  # imported lazily so local dev doesn't need boto3 configured
        return BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")  # Bedrock-hosted embedding model, same AWS account as the LLM calls

    # local, free fallback — no API key or AWS setup required
    from langchain_community.embeddings import HuggingFaceEmbeddings  # runs entirely on-device, good for iterating before Bedrock access is granted
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # small, fast, widely-used general-purpose embedding model


def build_index(services: list[dict], out_path: Path | str = DEFAULT_INDEX_PATH) -> FAISS:
    """Embed every service passage and persist a FAISS index to disk."""
    embeddings = get_embeddings()                                    # pick embedding backend (local or bedrock)

    texts = [to_document_text(s) for s in services]                  # one embeddable passage per service

    metadatas = [                                                     # structured fields kept alongside the vector for hard filtering later
        {
            "name": s["name"],
            "cost": s["cost_per_session"],
            "area": s["area"],
            "days": s["days"],
            "care_type": s["care_type"],
        }
        for s in services
    ]

    store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)  # builds the in-memory vector index from the passages
    store.save_local(str(out_path))                                   # persists the index + metadata to disk so retrieval_tool.py can load it without re-embedding
    return store


if __name__ == "__main__":
    services = load_services()                     # load the curated dataset
    build_index(services)                          # embed and persist the vector index
    print(f"Indexed {len(services)} services into {DEFAULT_INDEX_PATH}")  # simple confirmation for the terminal
