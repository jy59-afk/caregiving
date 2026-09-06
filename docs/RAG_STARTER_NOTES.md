# RAG Starter — Respite Navigator

Drop-in replacement for the flat JSON filter: embeds each service's
description into a vector index, then retrieves semantically before applying
hard budget/area filters.

## Setup

```bash
pip install langchain langchain-community faiss-cpu sentence-transformers
# add boto3 as well once you switch EMBEDDING_BACKEND=bedrock
```

## Files

- `data/services.sample.json` — schema example. Copy to `data/services.json`
  and replace with your real ~15+ curated services. The `description` field
  is what makes retrieval worthwhile — write it the way a caregiver would
  read it, not as a form field.
- `ingest.py` — run once (and whenever the dataset changes) to build the
  vector index at `data/vector_index/`.
- `retrieval_tool.py` — the `search_services(profile)` function your
  LangGraph/Claude Agent SDK Match & Rank node calls. Same signature as the
  old JSON-filter version, so the orchestration graph doesn't need to change.

## Usage

```bash
cp data/services.sample.json data/services.json   # or your real dataset
python ingest.py                                    # builds data/vector_index/
python retrieval_tool.py                            # smoke test with an example profile
```

## Switching embedding backends

Defaults to a free local model (`sentence-transformers/all-MiniLM-L6-v2`) so
you can iterate without AWS credentials. Once Bedrock access is set up:

```bash
export EMBEDDING_BACKEND=bedrock
python ingest.py     # re-run to rebuild the index with Titan embeddings — the two backends are not interchangeable, always rebuild after switching
```

## Evaluating retrieval quality

Don't skip this — it's the evidence that turns "we added RAG" into a graded,
defensible claim. Build a small labeled set:

```python
# eval_retrieval.py (sketch — add to your tests/ folder)
labeled_queries = [
    {"query": "dad gets confused, needs patient supervision during the day", "expected": "Sunshine Corner Day Centre"},
    {"query": "mum can't leave the house, needs help with mobility at home", "expected": "Golden Years Home Respite"},
    # add 8-10 more covering your real service variety
]

def recall_at_k(k=3):
    hits = 0
    for case in labeled_queries:
        results = search_services({"needs_description": case["query"], "budget": None, "area": None}, k=k)
        if any(r["name"] == case["expected"] for r in results):
            hits += 1
    return hits / len(labeled_queries)
```

Report `recall@3` on your evaluation slide alongside schema validation pass
rate and tool-call success rate — it's the retrieval-specific metric judges
will expect once you claim RAG.
