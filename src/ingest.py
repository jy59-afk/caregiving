"""
ingest.py — one-time (or whenever the dataset changes) script that turns the
curated respite-service records into a searchable vector index.

Run from the repo root:  python src/ingest.py
Rebuild whenever data/services.json changes, or after switching EMBEDDING_BACKEND.
"""

import json  # for loading the curated service records from disk + writing the index sidecar
from pathlib import Path  # for building filesystem paths that do not depend on the caller's working directory

from langchain_community.vectorstores import FAISS  # lightweight, zero-infra vector store — good fit for a few dozen/hundred records

from config import settings  # typed view of .env — embedding backend + Bedrock model id + region

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


LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # the on-device model name, referenced by the index sidecar too


def get_embeddings():
    """
    Return the embedding backend to use, switchable via `EMBEDDING_BACKEND` in
    `.env` so local/free development and the Bedrock demo build share the same
    ingestion script.

      local   -> sentence-transformers/all-MiniLM-L6-v2  (384-dim, on-device, free)
      bedrock -> settings.bedrock_embed_model            (default `cohere.embed-english-v3`,
                 1024-dim — recall@3 0.80 here vs Titan V2's 0.50)

    The query path (`retrieval_tool`) calls this too, so a query is always
    embedded with the same model the index was built with.
    """
    if settings.embedding_backend == "bedrock":
        from langchain_community.embeddings import BedrockEmbeddings  # lazy: local dev doesn't need boto3 configured
        return BedrockEmbeddings(
            model_id=settings.bedrock_embed_model,   # e.g. "cohere.embed-english-v3"
            region_name=settings.aws_region,         # must match where the model is enabled
        )

    # local, free fallback — no API key or AWS setup required
    from langchain_community.embeddings import HuggingFaceEmbeddings  # runs entirely on-device
    return HuggingFaceEmbeddings(model_name=LOCAL_EMBED_MODEL)


def _current_embedding_id() -> str:
    """A short string identifying the model behind `get_embeddings()`, for the
    index sidecar so a stale index (wrong dimension) is caught, not crashed on."""
    if settings.embedding_backend == "bedrock":
        return f"bedrock:{settings.bedrock_embed_model}"
    return f"local:{LOCAL_EMBED_MODEL}"


INDEX_META_NAME = "meta.json"  # sidecar next to the FAISS files, recording which model built the index


def _write_index_meta(out_path: Path, services: list[dict], dim: int) -> None:
    """Record the embedding model + vector dimension used to build this index,
    so `retrieval_tool` can refuse a stale index rather than crash on a
    dimension mismatch when `EMBEDDING_BACKEND` is changed without rebuilding."""
    (Path(out_path) / INDEX_META_NAME).write_text(
        json.dumps(
            {
                "embedding_backend": settings.embedding_backend,
                "embedding_id": _current_embedding_id(),
                "dim": dim,
                "count": len(services),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def build_index(services: list[dict], out_path: Path | str = DEFAULT_INDEX_PATH) -> FAISS:
    """Embed every service passage and persist a FAISS index (+ a meta sidecar) to disk."""
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
    out_path = Path(out_path)
    store.save_local(str(out_path))                                   # persists the index + metadata to disk so retrieval_tool.py can load it without re-embedding
    _write_index_meta(out_path, services, dim=store.index.d)          # record the model/dim so a stale index is caught later
    return store


if __name__ == "__main__":
    services = load_services()                     # load the curated dataset
    store = build_index(services)                  # embed and persist the vector index
    print(
        f"Indexed {len(services)} services into {DEFAULT_INDEX_PATH} "
        f"({_current_embedding_id()}, dim={store.index.d})"
    )
