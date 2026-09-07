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

import json  # to read the index sidecar written by ingest.build_index
from pathlib import Path  # for a working-directory-independent path to the persisted index

from langchain_community.vectorstores import FAISS  # same vector store class used to build the index

from ingest import (  # reuse the exact embedding backend + index location used at ingestion time
    DEFAULT_INDEX_PATH,
    INDEX_META_NAME,
    _current_embedding_id,
    get_embeddings,
)

_store: FAISS | None = None  # module-level cache; the index is loaded once on first use, not at import time


def _check_index_matches_config(index_path: Path) -> None:
    """
    Refuse a stale index up front. If `EMBEDDING_BACKEND` (or the Bedrock embed
    model) was changed without re-running `python src/ingest.py`, the persisted
    vectors have the wrong dimension and FAISS crashes deep in a C call on the
    first query. Compare the sidecar `meta.json` to the current config instead.
    """
    meta_file = index_path / INDEX_META_NAME
    if not meta_file.exists():
        return  # index built before sidecars existed — nothing to check, let it load
    want = _current_embedding_id()
    have = json.loads(meta_file.read_text(encoding="utf-8")).get("embedding_id")
    if have and have != want:
        raise RuntimeError(
            f"The vector index was built with embeddings '{have}' but the current "
            f"config wants '{want}'. Rebuild it:  python src/ingest.py"
        )


def _get_store(index_path: Path | str = DEFAULT_INDEX_PATH) -> FAISS:
    """
    Lazily load and cache the FAISS index.

    Loading on first call (rather than at import) means importing this module
    never fails just because `python src/ingest.py` hasn't been run yet — which
    matters for test collection and for the CI/eval harness.
    """
    global _store  # we mutate the module-level cache
    if _store is None:  # first call — build the cache
        index_path = Path(index_path)
        _check_index_matches_config(index_path)  # clear error on a stale index, not a C-level crash
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

    `k` is how many results to RETURN (top-4 by similarity after filtering).
    Retrieval itself pulls a much wider net — near the whole corpus — so the
    budget/area hard filters have every record to work with. Without that, an
    area filter like "Bedok" silently empties the shortlist whenever the Bedok
    records don't happen to land in the semantic top-k (common now the index is
    ~80% look-alike nursing-home passages). A large-corpus build would push the
    metadata filter into the index instead of over-fetching like this.

    profile is expected to include:
      - needs_description: str  (the free-text need extracted from the conversation)
      - budget: float | None
      - area: str | None
    """
    query = profile["needs_description"]                          # what the caregiver actually said about the care recipient's needs

    store = _get_store()                                          # load (once) the persisted vector index
    net = max(k * 8, 200)                                         # over-fetch: with ~76 records this is effectively "rank the whole corpus"
    candidates = store.similarity_search_with_score(query, k=net)

    filtered = _hard_filter(candidates, profile)                  # enforce budget/area as hard constraints against the FULL ranked list

    top4 = sorted(filtered, key=lambda pair: pair[1])[:4]          # keep the 4 closest semantic matches that also passed the hard filter

    return [
        {
            "name": doc.metadata["name"],           # service name, for the recommendation list
            "grounding_passage": doc.page_content,  # the actual retrieved text — pass this into the Explain & Draft node so its
                                                     # "why this fits" reasoning is grounded in real service content, not invented
            "similarity_score": float(distance),    # kept for logging / evaluation (recall@k, etc.), not shown to the caregiver
        }
        for doc, distance in top4
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
