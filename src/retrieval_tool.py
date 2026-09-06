"""
retrieval_tool.py — the tool the agent calls at match time. Replaces a flat
JSON filter with hybrid retrieval: vector similarity for semantic fit
(care type, atmosphere, programme details), then hard filtering for
non-negotiable constraints (budget, area) that embeddings can't reliably judge.

This keeps the same function signature (`search_services(profile) -> list`)
that the LangGraph / Claude Agent SDK node already calls, so swapping this in
for the old JSON filter doesn't require changing the orchestration graph —
only the tool's internals change.
"""

from pathlib import Path  # for a working-directory-independent path to the persisted index

from langchain_community.vectorstores import FAISS  # same vector store class used to build the index

from ingest import DEFAULT_INDEX_PATH, get_embeddings  # reuse the exact embedding backend + index location used at ingestion time — a mismatch here silently degrades results

_store: FAISS | None = None  # module-level cache; the index is loaded once on first use, not at import time


def _get_store(index_path: Path | str = DEFAULT_INDEX_PATH) -> FAISS:
    """
    Lazily load and cache the FAISS index.

    Loading on first call (rather than at import) means importing this module
    never fails just because `python src/ingest.py` hasn't been run yet — which
    matters for test collection and for the CI/eval harness.
    """
    global _store  # we mutate the module-level cache
    if _store is None:  # first call — build the cache
        embeddings = get_embeddings()  # embedding model init is the expensive step, so do it at most once per process
        _store = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,  # required by FAISS's loader; safe here because we only ever load an index we built ourselves
        )
    return _store  # cached store on every subsequent call


def _hard_filter(candidates: list[tuple], profile: dict) -> list[tuple]:
    """
    Drop any semantically-similar candidate that fails a non-negotiable
    constraint. Budget and area are treated as hard filters because a
    caregiver who said "$40 max" does not want a $120/session suggestion,
    however good the semantic match.
    """
    out = []                                          # collects candidates that survive the hard constraints
    for doc, distance in candidates:                  # FAISS returns (Document, distance) pairs, lower distance = closer match
        meta = doc.metadata                           # structured fields stored alongside the embedded passage

        if profile.get("budget") is not None and meta["cost"] > profile["budget"]:
            continue                                  # over budget — reject regardless of semantic fit

        if profile.get("area") and profile["area"].lower() not in meta["area"].lower():
            continue                                  # simple substring area match; swap in a real distance/geocoding check if you have one

        out.append((doc, distance))                   # candidate passes every hard constraint
    return out


def search_services(profile: dict, k: int = 8) -> list[dict]:
    """
    Hybrid retrieval entry point called by the agent's Match & Rank node.

    profile is expected to include:
      - needs_description: str  (the free-text need extracted from the conversation)
      - budget: float | None
      - area: str | None
    """
    query = profile["needs_description"]                          # what the caregiver actually said about the care recipient's needs

    store = _get_store()                                          # load (once) the persisted vector index
    candidates = store.similarity_search_with_score(query, k=k)   # cast a wider net semantically first (k=8), filter down after

    filtered = _hard_filter(candidates, profile)                  # enforce budget/area as hard constraints

    top3 = sorted(filtered, key=lambda pair: pair[1])[:3]          # keep the 3 closest semantic matches that also passed the hard filter

    return [
        {
            "name": doc.metadata["name"],           # service name, for the recommendation list
            "grounding_passage": doc.page_content,  # the actual retrieved text — pass this into the Explain & Draft node so its
                                                     # "why this fits" reasoning is grounded in real service content, not invented
            "similarity_score": float(distance),    # kept for logging / evaluation (recall@k, etc.), not shown to the caregiver
        }
        for doc, distance in top3
    ]


if __name__ == "__main__":
    # Quick manual smoke test — run after `python src/ingest.py` has built the index.
    example_profile = {
        "needs_description": "my dad gets confused and repeats himself, needs someone patient with him during the day while I work",
        "budget": 90,        # rules out the ~S$120/day nursing-home respite, keeps the day centres
        "area": "Toa Payoh",  # substring-matched against each service's area
    }
    matches = search_services(example_profile)  # hybrid: semantic retrieve, then hard-filter on budget + area
    if not matches:
        print("No services matched — loosen the budget or area, or run `python src/ingest.py` first.")
    for match in matches:
        print(match["name"], "-", round(match["similarity_score"], 3))  # lower score = closer semantic match
